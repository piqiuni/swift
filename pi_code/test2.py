import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type, inference_stream_vllm,
)
from swift.utils import seed_everything
from swift.tuners import Swift

ckpt_dir = '/home/ldl/pi_code/swift/output/qwen-vl/v6-20240421-110548/checkpoint-71' #加载模型路径
model_type = ModelType.qwen_vl
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True) #加载微调后的模型
template = get_template(template_type, tokenizer)
seed_everything(42)

query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': '这是什么？'},
])
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')

request_list = [{'query': 'who are you?'}]

gen = inference(model, template, request_list)
query = request_list[0]['query']
print(f'query: {query}\nresponse: ', end='')
print_idx = 0
for resp_list in gen:
    response = resp_list[0]['response']
    print(response[print_idx:], end='', flush=True)
    print_idx = len(response)
print()


response, history = inference(model, template, query)
print(f'template: {template}')
print(f'template.prompt: {template.prompt}')
print(f'query: {query}')
print(f'response: {response}')

print("----------------")



