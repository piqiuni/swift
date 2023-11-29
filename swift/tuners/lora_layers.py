# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
import math
import re
import warnings
from itertools import chain
from typing import Dict, List

import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import Conv2d as _Conv2d
from peft.tuners.lora import Embedding as _Embedding
from peft.tuners.lora import Linear as _Linear
from peft.tuners.lora import LoraLayer
from peft.tuners.lora import LoraModel as _LoraModel
from peft.utils import get_auto_gptq_quant_linear, get_quantization_config
from transformers import Conv1D

from swift import get_logger
from .utils import ActivationMixin

logger = get_logger()


def is_auto_gptq_available():
    try:
        return peft.import_utils._is_auto_gptq_available()
    except ImportError as e:
        logger.warn(e)
        return False


peft.import_utils._is_auto_gptq_available = peft.import_utils.is_auto_gptq_available
peft.import_utils.is_auto_gptq_available = is_auto_gptq_available


class LoRAActivationMixin(ActivationMixin):

    @property
    def active_adapters(self):
        return self.get_activated_adapters()

    @property
    def active_adapter(self) -> str:
        return self.get_activated_adapters()

    def set_adapter(self, adapter_names):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    self.set_activation(key, True)
                    layer.requires_grad_(True)
                else:
                    self.set_activation(key, False)
                    layer.requires_grad_(False)

    def merge(self, *args, **kwargs):
        if not self.unique_thread:
            raise AssertionError(
                'Merge is unsupported in multiple thread, '
                'please set `USE_UNIQUE_THREAD=1` in env variable to merge LoRA.'
            )
        return super().merge(*args, **kwargs)


if is_bnb_available():
    import bitsandbytes as bnb
    from peft.tuners.lora import Linear8bitLt as _Linear8bitLt

    class Linear8bitLt(LoRAActivationMixin, _Linear8bitLt):

        def __init__(
            self,
            adapter_name,
            base_layer,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            super(ActivationMixin,
                  self).__init__(adapter_name, base_layer, r, lora_alpha,
                                 lora_dropout, **kwargs)
            super(Linear8bitLt, self).__init__()


if is_bnb_4bit_available():
    from peft.tuners.lora import Linear4bit as _Linear4bit

    class Linear4bit(LoRAActivationMixin, _Linear4bit):

        def __init__(
            self,
            adapter_name,
            base_layer,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            super(Linear4bit, self).__init__()
            super(ActivationMixin,
                  self).__init__(adapter_name, base_layer, r, lora_alpha,
                                 lora_dropout, **kwargs)


if is_auto_gptq_available():
    from peft.tuners.lora import QuantLinear as _QuantLinear

    class QuantLinear(LoRAActivationMixin, _QuantLinear):

        def __init__(
            self,
            *args,
            use_qa_lora=False,
            group_size=None,
            **kwargs,
        ):
            super(ActivationMixin, self).__init__(*args, **kwargs)
            super(QuantLinear, self).__init__()
            self.group_size = group_size
            self.use_qa_lora = use_qa_lora
            if self.use_qa_lora:
                assert self.group_size is not None, 'To use qa_lora you need to pass in the `group_size` param.'
            if self.use_qa_lora:
                self.qa_pool = torch.nn.AvgPool1d(
                    self.group_size
                )  # using pooling layer to conduct sum operation

        def forward(self, x: torch.Tensor):
            # note: logic differs from default Linear because merging is not supported
            result = self.quant_linear_module(x)

            if self.disable_adapters:
                return result

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                if self.use_qa_lora:
                    x = self.qa_pool(x) * self.group_size
                output = lora_B(lora_A(dropout(x)))
                if requires_conversion:
                    output = output.to(expected_dtype)
                output = output * scaling
                result += output
            return result


class Embedding(LoRAActivationMixin, _Embedding):

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super(Embedding, self).__init__()
        super(ActivationMixin, self).__init__(*args, **kwargs)


class Linear(LoRAActivationMixin, _Linear):

    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__()
        super(ActivationMixin, self).__init__(*args, **kwargs)


class Conv2d(LoRAActivationMixin, _Conv2d):

    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__()
        super(ActivationMixin, self).__init__(*args, **kwargs)


class LoraModel(_LoraModel):

    prefix: str = 'lora_'

    def __init__(self, model, config, adapter_name):
        if config is not None:
            super().__init__(model, config, adapter_name)
        else:
            nn.Module.__init__(self)
            self.model = model

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(
            chain(lora_config.rank_pattern.keys(),
                  lora_config.alpha_pattern.keys()))
        target_name_key = next(
            filter(
                lambda key: re.match(f'.*\.{key}$', current_key),  # noqa
                pattern_keys),
            current_key)

        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key,
                                              lora_config.lora_alpha)
        bias = hasattr(target, 'bias') and target.bias is not None
        kwargs = {
            'r': r,
            'lora_alpha': alpha,
            'lora_dropout': lora_config.lora_dropout,
            'fan_in_fan_out': lora_config.fan_in_fan_out,
            'init_lora_weights': lora_config.init_lora_weights,
        }
        kwargs['loaded_in_8bit'] = optional_kwargs.pop('loaded_in_8bit', False)
        kwargs['loaded_in_4bit'] = optional_kwargs.pop('loaded_in_4bit', False)
        kwargs['bias'] = bias

        quantization_config = get_quantization_config(
            self.model, method='gptq')
        if quantization_config is not None:
            kwargs['gptq_quantization_config'] = quantization_config

        # TODO: better deal with that
        if isinstance(target, LoraLayer) and isinstance(
                target, torch.nn.Conv2d):
            target.update_layer_conv2d(
                adapter_name,
                r,
                alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
        elif isinstance(target, LoraLayer) and isinstance(
                target, torch.nn.Embedding):
            target.update_layer_embedding(
                adapter_name,
                r,
                alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

        elif isinstance(target, LoraLayer):
            target.update_layer(
                adapter_name,
                r,
                alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name,
                                                 target, **kwargs)
            if new_module is not None:
                if adapter_name != self.active_adapter:
                    # adding an additional adapter: it is not automatically trainable
                    new_module.requires_grad_(False)
                self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        gptq_quantization_config = kwargs.get('gptq_quantization_config', None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(
            gptq_quantization_config)

        loaded_in_8bit = kwargs.pop('loaded_in_8bit', False)
        loaded_in_4bit = kwargs.pop('loaded_in_4bit', False)
        bias = kwargs.pop('bias', False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update({
                'has_fp16_weights': target.state.has_fp16_weights,
                'memory_efficient_backward':
                target.state.memory_efficient_backward,
                'threshold': target.state.threshold,
                'index': target.index,
            })
            new_module = Linear8bitLt(adapter_name, target, **eightbit_kwargs)
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(
                target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update({
                'compute_dtype': target.compute_dtype,
                'compress_statistics': target.weight.compress_statistics,
                'quant_type': target.weight.quant_type,
            })
            new_module = Linear4bit(adapter_name, target, **fourbit_kwargs)
        elif AutoGPTQQuantLinear is not None and isinstance(
                target, AutoGPTQQuantLinear):
            new_module = QuantLinear(adapter_name, target, **kwargs)
            target.weight = target.qweight
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop('fan_in_fan_out', None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features,
                                   **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels,
                                kernel_size, stride, padding, **kwargs)
        elif lora_config.use_merged_linear:
            new_module = MergedLinear(
                adapter_name,
                target,
                bias=bias,
                enable_lora=lora_config.enable_lora,
                **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs['fan_in_fan_out']:
                    warnings.warn(
                        'fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. '
                        'Setting fan_in_fan_out to False.')
                    kwargs[
                        'fan_in_fan_out'] = lora_config.fan_in_fan_out = False
                new_module = Linear(
                    adapter_name,
                    in_features,
                    out_features,
                    bias=bias,
                    **kwargs)
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(
                        target.weight, 'ds_shape') else target.weight.shape)
                kwargs['is_target_conv_1d_layer'] = True
                if not kwargs['fan_in_fan_out']:
                    warnings.warn(
                        'fan_in_fan_out is set to False but the target module is `Conv1D`. '
                        'Setting fan_in_fan_out to True.')
                    kwargs[
                        'fan_in_fan_out'] = lora_config.fan_in_fan_out = True
                new_module = Linear(
                    adapter_name,
                    in_features,
                    out_features,
                    bias=bias,
                    **kwargs)
            else:
                logger.debug(
                    f'Target module {target} is not supported. Currently, only the following modules are supported: '
                    '`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`.'
                )
                new_module = None

        return new_module


class LoRALayer(ActivationMixin):

    def __init__(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        super().__init__()
        self.adapter_name = adapter_name
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        if not self._unique_thread:
            self.merge_weights = False


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(self,
                 adapter_name: str,
                 base_layer: nn.Linear,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.,
                 enable_lora: List[bool] = [False],
                 fan_in_fan_out: bool = False,
                 merge_weights: bool = True,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs):
        nn.Linear.__init__(
            self,
            base_layer.in_features,
            base_layer.out_features,
            bias=bias,
            device=device,
            dtype=dtype)
        LoRALayer.__init__(
            self,
            adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights)
        assert base_layer.out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.base_layer = base_layer
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros(
                    (r * sum(enable_lora), base_layer.in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (base_layer.out_features // len(enable_lora)
                     * sum(enable_lora),
                     r)))  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (base_layer.out_features, ),
                dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        self.weight = self.base_layer.weight
        if getattr(self.base_layer, 'bias', None) is not None:
            self.bias = self.base_layer.bias
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora)).squeeze(0)
        return T(self.zero_pad(delta_w))

    def merge(self, **kwargs):
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                self.weight.data += self.merge_AB() * self.scaling

    def unmerge(self, **kwargs):
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                self.weight.data -= self.merge_AB() * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged or not self.is_activated(self.adapter_name):
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                x_dtype = x.dtype
                x = x.to(self.lora_A.dtype)
                result += self.lora_dropout(x) @ T(
                    self.merge_AB().T) * self.scaling
                result = result.to(x_dtype)
            return result


def mark_lora_as_trainable(model: nn.Module,
                           adapter_name: str,
                           bias: str = 'none') -> None:
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for n, m in model.named_modules():
            if f'loramodule_{adapter_name}' in n and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(state_dict,
                    adapter_name: str,
                    bias: str = 'none') -> Dict[str, torch.Tensor]:
    if bias == 'none':
        to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k}
    elif bias == 'all':
        to_return = {
            k: state_dict[k]
            for k in state_dict if 'lora_' in k or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        for k in state_dict:
            if 'lora_' in k:
                to_return[k] = state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    return {
        k: v
        for k, v in to_return.items()
        if (('lora_' in k and adapter_name in k) or ('bias' in k))
    }
