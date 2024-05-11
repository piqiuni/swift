# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
from pi_code.uts import default_system

use_one_gpu = False
if use_one_gpu:
    # raise "OOM Error"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['NPROC_PER_NODE'] = '1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['NPROC_PER_NODE'] = '1'

os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main, merge_lora
)

model_type = ModelType.internvl_chat_v1_5

# mini
# custom_train_dataset_path = '~/pi_code/swift/pi_code/mini_trainning_llama.json'
# full
custom_train_dataset_path = '~/pi_code/swift/pi_code/trainning_llama.json'
custom_train_dataset_path = '~/pi_code/swift/pi_code/history_trainning_llama.json'

sft_args = SftArguments(
    model_type=model_type,
    train_dataset_sample=-1,
    # sft_type='lora',
    # quantization_bit=4,
    # deepspeed='default-zero2',
    # dataset='coco-mini-en',
    dataset = custom_train_dataset_path,
    # resume_from_checkpoint = 'ckp_output/qwen-vl/v10-20240429-172025/checkpoint-3644',
    system = default_system,
    num_train_epochs = 3,
    eval_steps = 200,
    batch_size=1,
    max_length=4096,
    output_dir='./ckp_output')
# assert os.path.exists(sft_args.output_dir)
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()

# infer_args = InferArguments(
#     ckpt_dir=best_model_checkpoint,
#     load_dataset_config=True,
#     val_dataset_sample=10)
# # merge_lora(infer_args, device_map='cpu')
# result = infer_main(infer_args)
# torch.cuda.empty_cache()

# app_ui_main(infer_args)
