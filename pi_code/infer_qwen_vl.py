import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
from swift.tuners import Swift
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json

now = datetime.now()
ckpt_dir = '/home/ldl/pi_code/swift/output/qwen-vl/v6-20240421-110548/checkpoint-71' #加载模型路径
infer_dataset_path = '/home/ldl/pi_code/swift/pi_code/mini_trainning_llama.json' #加载数据集路径
# infer_dataset_path = '/home/ldl/pi_code/swift/pi_code/test_val_llama.json'

file_name = f"output_{now.strftime('%m%d_%H%M')}.json"
save_path = os.path.join('./output', file_name)

model_type = ModelType.qwen_vl
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True) #加载微调后的模型
template = get_template(template_type, tokenizer)
seed_everything(42)

# 读取 JSON 文件
with open(infer_dataset_path) as file:
    data = json.load(file)

output_data = []
# 循环提取每个段落的 conversations[0][value] 值
for paragraph in data:
    ids = paragraph["conversations"][0]["id"]
    value = paragraph["conversations"][0]["value"]
    lines = value.split("\n")
    last_line = lines[-1].split("/n")[-1]
    response, history = inference(model, template, value)

    # print(f'id: {ids}')
    # print(f'question: {last_line}') 
    # print(f'answer: {response}') 

    data = {
    "id": ids,
    "question": last_line,
    "answer": response
}
    output_data.append(data)

with open(save_path, 'w') as file:
    json.dump(output_data, file, indent=4)
print(f"Done, save to {save_path}")




