import os


from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['NPROC_PER_NODE'] = '1'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.llm.utils.utils import get_length
import yaml
from swift.utils import seed_everything
from swift.tuners import Swift
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json
from pi_code.uts import process_comment_question, default_system

now = datetime.now()
model_type = ModelType.internlm_xcomposer2_7b_chat
if model_type == None:
    ckpt_dir = '' #加载模型路径
elif model_type == ModelType.internlm_xcomposer2_7b_chat:
    max_his_length = 60
    ckpt_dir = '/home/ldl/pi_code/swift/ckp_output/internlm-xcomposer2-7b-chat/v21-20240524-212353/checkpoint-23490'
    

use_mini_data = False
file_name = f"output_{model_type}_{now.strftime('%m%d_%H%M')}.json"
if use_mini_data:
    infer_dataset_path = '/home/ldl/pi_code/swift/pi_code/mini_trainning_llama.json' #加载数据集路径
    file_name = f"mini_output_{model_type}_{now.strftime('%m%d_%H%M')}.json"
else:
    infer_dataset_path = '/home/ldl/pi_code/swift/pi_code/val_llama.json'

save_folder = './pi_code/output'
save_path = os.path.join(save_folder, file_name)
config_path = save_path.replace('.json', '_config.yaml')
config_data = {}
config_data["model_type"] = str(model_type)
config_data["ckpt_dir"] = ckpt_dir
config_data["infer_dataset_path"] = infer_dataset_path
config_data["save_path"] = save_path

template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True) #加载微调后的模型
template = get_template(template_type, tokenizer)
seed_everything(42)
print("Got model")
template.default_system = default_system
print(f"template.default_system: {template.default_system}")
# print(model.config)
# print(model.generation_config)
# raise
model.config.max_length = 4096*1# 4096
# model.generation_config.max_length = 256
model.generation_config.max_new_tokens = 256
# raise

# 读取 JSON 文件
with open(infer_dataset_path) as file:
    data = json.load(file)

output_file = []
# 循环提取每个段落的 conversations[0][value] 值
history = []
last_id_head = None
max_total_len = 0
print(len(data))
for i in tqdm(range(len(data))):
# for i in tqdm(range(len(data[:500]))):
    paragraph = data[i]
    ids = paragraph["conversations"][0]["id"]
    id_head = ids.split("_")[:-1]
    if id_head != last_id_head:
        history = []
    last_id_head = id_head
    value = paragraph["conversations"][0]["value"]
    lines = value.split("\n")
    raw_question = lines[-1]
    new_question = process_comment_question(raw_question)
    lines[-1] = new_question
    value = "".join(lines)
    if model_type == None:
        response, history = inference(model, template, value)
    elif model_type == ModelType.internlm_xcomposer2_7b_chat:
        his_length = len(history)
        start_index = max(0, his_length - max_his_length)
        history = history[start_index:]
        
        # print(history)
        # print(value)
        len_his = get_length(model, template, str(history))
        len_val = get_length(model, template, value)
        len_total = len_his + len_val
        print(f"token len: history:{len_his}, value:{len_val}, total:{len_total}")
        max_total_len = max(max_total_len, len_total)
        
        response, _ = inference(model, template, value, history)
        print(f"raw_question: {raw_question}")
        print(f"response: {response.strip()}")
        print("-------")
        
        qa = [new_question, response.strip()]
        history.append(qa)
    # [['Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>\n距离各城市多远？', '马路边距离马路边14公里；阳江边距离马路边62公里；广州边距离马路边293公里。'], ['距离最远的城市是哪？', '距离最远的城市是广州，距离马路边293公里。']]
    
    output_data = {
    "id": ids,
    "question": raw_question,
    "answer": response
}
    output_file.append(output_data)

with open(save_path, 'w') as file:
    json.dump(output_file, file, indent=4)
with open(config_path, 'w') as f:
    yaml.dump(config_data, f, default_flow_style=False)
print(f"Done, save to {save_path}")

print(f"max_total_len: {max_total_len}")




