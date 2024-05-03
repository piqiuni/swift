
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['NPROC_PER_NODE'] = '2'

os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'


import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main, merge_lora
)
from tqdm import tqdm

model_type = ModelType.qwen_vl_chat

# infer
use_mini_data = False

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
    # resume_from_checkpoint = 'ckp_output/qwen-vl/v10-20240429-172025/checkpoint-3644',
    # save_only_model = False,
    max_length=4096,
    output_dir='./ckp_output')
# assert os.path.exists(sft_args.output_dir)
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()


# Infer
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.llm.utils.utils import get_length

from swift.utils import seed_everything
from swift.tuners import Swift
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json

now = datetime.now()
os.environ['NPROC_PER_NODE'] = '1'


# if model_type == ModelType.qwen_vl:
#     # ckpt_dir = '/home/ldl/pi_code/swift/ckp_output/qwen-vl/v11-20240430-040911/checkpoint-4200' #加载模型路径
#     ckpt_dir = '/home/ldl/pi_code/swift/ckp_output/qwen-vl/v12-20240502-170251/checkpoint-71'
# elif model_type == ModelType.qwen_vl_chat:
#     ckpt_dir = '/home/ldl/pi_code/swift/ckp_output/qwen-vl-chat/v2-20240502-164517/checkpoint-71'

ckpt_dir = best_model_checkpoint


file_name = f"output_{model_type}_{now.strftime('%m%d_%H%M')}.json"
if use_mini_data:
    infer_dataset_path = '/home/ldl/pi_code/swift/pi_code/mini_trainning_llama.json' #加载数据集路径
    file_name = f"mini_output_{model_type}_{now.strftime('%m%d_%H%M')}.json"
else:
    infer_dataset_path = '/home/ldl/pi_code/swift/pi_code/val_llama.json'

save_folder = './pi_code/output'
save_path = os.path.join(save_folder, file_name)
# assert os.path.exists(save_folder)

template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True) #加载微调后的模型
template = get_template(template_type, tokenizer)
seed_everything(42)
print("Got model")
print(f"template.default_system: {template.default_system}")
# print(model.config)
model.config.seq_length = 4096
# raise

# 读取 JSON 文件
with open(infer_dataset_path) as file:
    data = json.load(file)

output_file = []
# 循环提取每个段落的 conversations[0][value] 值
history = []
last_id_head = None
# for i in tqdm(range(len(data[:]))):
for i in tqdm(range(len(data[:500]))):
    paragraph = data[i]
    ids = paragraph["conversations"][0]["id"]
    id_head = ids.split("_")[:-1]
    if id_head != last_id_head:
        history = []
    last_id_head = id_head
    value = paragraph["conversations"][0]["value"]
    lines = value.split("\n")
    last_line = lines[-1].split("/n")[-1]
    # value = "\n".join(lines[:-1])
    # print(value)
    if model_type == ModelType.qwen_vl:
        response, history = inference(model, template, value)
    elif model_type == ModelType.qwen_vl_chat:
        his_length = len(history)
        max_his_length = 60
        start_index = max(0, his_length - max_his_length)
        history = history[start_index:]
        # print(history)
        # print(value)
        # print(f"token len:{get_length(model, template, str(history))}, {get_length(model, template, value)}")
        response, _ = inference(model, template, value, history)
        # print(response)
        # print("-------")
        qa = [last_line, response]
        history.append(qa)
        # print(value)
        # print(response)
    # [['Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>\n距离各城市多远？', '马路边距离马路边14公里；阳江边距离马路边62公里；广州边距离马路边293公里。'], ['距离最远的城市是哪？', '距离最远的城市是广州，距离马路边293公里。']]
    
    output_data = {
    "id": ids,
    "question": last_line,
    "answer": response
}
    output_file.append(output_data)

with open(save_path, 'w') as file:
    json.dump(output_file, file, indent=4)
print(f"Done, save to {save_path}")

print(f'best_model_checkpoint: {best_model_checkpoint}')
