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
    os.environ['NPROC_PER_NODE'] = '2'

# os.environ['NCCL_IB_DISABLE'] = '1'
# os.environ['NCCL_P2P_DISABLE'] = '1'


import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main, merge_lora
)

model_type = ModelType.internlm_xcomposer2_7b_chat

# mini
# custom_train_dataset_path = '~/pi_code/swift/pi_code/mini_trainning_llama.json'
# full
# custom_train_dataset_path = '~/pi_code/swift/pi_code/trainning_llama.json'
custom_train_dataset_path = '~/pi_code/swift/pi_code/history_trainning_llama.json'

sft_args = SftArguments(
    model_type=model_type, # model_type = ModelType.internlm_xcomposer2_7b_chat
    train_dataset_sample=-1,
    # sft_type='lora',
    # quantization_bit=4,
    # deepspeed='default-zero2',
    # dataset='coco-mini-en',
    dataset = custom_train_dataset_path, # '~/pi_code/swift/pi_code/history_trainning_llama.json'
    resume_from_checkpoint = 'ckp_output/internlm-xcomposer2-7b-chat/v21-20240524-212353/checkpoint-23490',
    system = default_system,  #'You are an experienced driver who can answer questions based on perceptual images. Reply with only English'
    logging_steps = 10,
    num_train_epochs = 15,
    eval_steps = 200,
    batch_size=1,
    max_length=4096,
    output_dir='./ckp_output')
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
