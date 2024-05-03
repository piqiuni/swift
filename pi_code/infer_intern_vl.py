import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
from swift.llm.utils.utils import get_length
import torch

model_type = ModelType.internvl_chat_v1_5
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = [
    'samples/CAM_FRONT/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404.jpg',
    'samples/CAM_FRONT_LEFT/n008-2018-09-18-13-10-39-0400__CAM_FRONT_LEFT__1537291010604799.jpg',
    'samples/CAM_FRONT_RIGHT/n008-2018-09-18-13-10-39-0400__CAM_FRONT_RIGHT__1537291010620482.jpg',
    'samples/CAM_BACK/n008-2018-09-18-13-10-39-0400__CAM_BACK__1537291010637558.jpg',
    'samples/CAM_BACK_LEFT/n008-2018-09-18-13-10-39-0400__CAM_BACK_LEFT__1537291010647405.jpg',
    'samples/CAM_BACK_RIGHT/n008-2018-09-18-13-10-39-0400__CAM_BACK_RIGHT__1537291010628113.jpg']

query = 'What actions taken by the ego vehicle can lead to a collision with <c1,CAM_BACK,1088.3,497.5>?'
token_length = get_length(model, template, str(query + str(images)))
images_length = get_length(model, template, str(images))
print(f"token_length:{token_length}; images_length:{images_length}")
input("input to start")
response, history = inference(model, template, query, images=images) # chat with image
print(f'query: {query}')
print(f'response: {response}')

input("input")

# 流式
query = 'In this scenario, what are safe actions to take for the ego vehicle?'
gen = inference_stream(model, template, query, history) # chat without image
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, history in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(f'history: {history}')

"""
query: 距离各城市多远？
response: 这张图片显示的是一个路标，上面标示了三个目的地及其距离：

- 马踏（Mata）：14公里
- 阳江（Yangjiang）：62公里
- 广州（Guangzhou）：293公里

这些距离是按照路标上的指示来计算的。
query: 距离最远的城市是哪？
response: 根据这张图片，距离最远的城市是广州（Guangzhou），距离为293公里。
history: [['距离各城市多远？', '这张图片显示的是一个路标，上面标示了三个目的地及其距离：\n\n- 马踏（Mata）：14公里\n- 阳江（Yangjiang）：62公里\n- 广州（Guangzhou）：293公里\n\n这些距离是按照路标上的指示来计算的。 '], ['距离最远的城市是哪？', '根据这张图片，距离最远的城市是广州（Guangzhou），距离为293公里。 ']]
"""