# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['NPROC_PER_NODE'] = '2'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main, merge_lora
)

model_type = ModelType.qwen_vl
# mini
# custom_train_dataset_path = '~/pi_code/swift/pi_code/mini_trainning_llama.json'
# full
custom_train_dataset_path = '~/pi_code/swift/pi_code/trainning_llama.json'

sft_args = SftArguments(
    model_type=model_type,
    train_dataset_sample=-1,
    custom_train_dataset_path=custom_train_dataset_path,
    num_train_epochs = 3,
    eval_steps = 200,
    output_dir='./output')
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()

infer_args = InferArguments(
    ckpt_dir=best_model_checkpoint,
    load_dataset_config=True,
    val_dataset_sample=10)
# merge_lora(infer_args, device_map='cpu')
result = infer_main(infer_args)
torch.cuda.empty_cache()

app_ui_main(infer_args)