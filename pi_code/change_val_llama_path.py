import json


file_path = "/home/ldl/pi_code/swift/pi_code/val_llama.json"

# 读取 JSON 文件
with open(file_path, 'r') as f:
    data = json.load(f)

# 定义新路径
old_path = '/home/ldl/pi_code/DriveLM/challenge/llama_adapter_v2_multimodal7b/data/nuscenes/samples'
new_path = '/home/ldl/pi_code/swift/samples'

# 遍历字典,替换路径
for item in data:
    for conv in item['conversations']:
        conv['value'] = conv['value'].replace(old_path, new_path)

# 将修改后的数据写回到新的 JSON 文件
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)