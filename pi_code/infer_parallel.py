import os
import torch
from tqdm import tqdm

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
from multiprocessing import Pool

def test_data(args):
    model, template,  data_split = args
    output_file = []
    first_data = data_split[0]
    last_data = data_split[-1]
    info = [first_data["conversations"][0]["id"], last_data["conversations"][0]["id"]]
    output_file.append(info)
    return output_file

def process_data(args):
    model, template,  data_split = args
    output_file = []
    history = []
    last_id_head = None
    print(f"Start process data: {len(data_split)}")
    for i in tqdm(range(len(data_split))):
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
        print(value)
        response, history = inference(model, template, value, history)
        print(response)
        qa = [new_question, response.strip()]
        history.append(qa)
        
        output_data = {
            "id": ids,
            "question": raw_question,
            "answer": response
        }
        output_file.append(output_data)
    
    return output_file


def start_infer(args):
    config_data, data_split, device_id = args
    model_type = config_data["model_type"]
    ckpt_dir = config_data["ckpt_dir"]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
    model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
    model.config.max_length = 4096 * 1
    model.generation_config.max_new_tokens = 256
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, tokenizers[0])
    output_file = []
    history = []
    last_id_head = None
    print(f"Start process data: {len(data_split)}")
    for i in tqdm(range(len(data_split))):
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
        print(value)
        response, history = inference(model, template, value, history)
        print(response)
        qa = [new_question, response.strip()]
        history.append(qa)
        
        output_data = {
            "id": ids,
            "question": raw_question,
            "answer": response
        }
        output_file.append(output_data)
    
    return output_file
    

now = datetime.now()
model_type = ModelType.internlm_xcomposer2_7b_chat
if model_type == None:
    ckpt_dir = ''
elif model_type == ModelType.internlm_xcomposer2_7b_chat:
    max_his_length = 60
    ckpt_dir = '/home/ldl/pi_code/swift/ckp_output/internlm-xcomposer2-7b-chat/v21-20240524-212353/checkpoint-23490'
else:
    raise ValueError(f"Invalid model_type: {model_type}")

use_mini_data = False
file_name = f"output_{model_type}_{now.strftime('%m%d_%H%M')}.json"
if use_mini_data:
    infer_dataset_path = '/home/ldl/pi_code/swift/pi_code/mini_trainning_llama.json'
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



# Load the model on 4 different GPUs
device_ids = [0, 1, 2, 3]
num_workers = len(device_ids)


# Read the JSON file
with open(infer_dataset_path) as file:
    data = json.load(file)
data = data[:100]

data_length= len(data)
split_index = [0]
seed_everything(42)
for i in range(len(device_ids)-1):
    end_index = int(data_length / len(device_ids) * (i + 1))
    end_data = data[end_index]
    end_index -= int(end_data["conversations"][0]["id"].split("_")[-1])
    split_index.append(end_index)
    print(data[end_index]["conversations"][0]["id"])
split_index.append(data_length-1)
print(data_length, split_index)
data_splits = [data[split_index[i]:split_index[i+1]] for i in range(len(device_ids))]
print([len(data_split) for data_split in data_splits])


with Pool(processes=num_workers) as pool:
    args_list = []
    for i in device_ids:
        args_list.append([config_data, data_splits[i], i])
    results = pool.map(start_infer, args_list)


output_files = [result for result in results]

    

with open(save_path, 'w') as file:
    json.dump(output_files, file, indent=4)
with open(config_path, 'w') as f:
    yaml.dump(config_data, f, default_flow_style=False)
print(f"Done, save to {save_path}")

# print(f"max_total_len: {max_total_len}")