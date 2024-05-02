import os
import json
import numpy as np
import torch
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.llm.utils.utils import get_length

from tqdm import tqdm
from transformers import (GenerationConfig, PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase, StoppingCriteriaList,
                          TextStreamer, trainer)
from swift.tuners import Swift




if __name__ == "__main__":
    filepath = "/home/ldl/pi_code/DriveLM/challenge/data/v1_1_train_nus.json"
    """_summary_
    length of queries: 4072
    ave_query:92.81827111984283, max_query:160
    ave_qa_length:4092.7973968565816, max_qa_length:7157, np.percentile(a,95):5535.349999999999, np.percentile(a,99):5925.0
    ave_q_length:2590.90594302554, max_q_length:5023
    ave_a_length:1437.0606581532415, max_a_length:2831
    ave_img_length:1618.0, max_img_length:1618
    """
    # 每个问答平均50，max_length=4096->80对
    # 滚动窗口处理，保留前80个
    
    # filepath = "/home/ldl/pi_code/DriveLM/challenge/v1_1_val_nus_q_only.json"
    """_summary_
    length of queries: 799
    ave_query:19.374217772215268, max_query:23
    ave_qa_length:773.5719649561952, max_qa_length:913, np.percentile(a,95):861.1999999999998, np.percentile(a,99):887.02
    ave_q_length:760.4868585732165, max_q_length:897
    ave_a_length:19.374217772215268, max_a_length:23
    ave_img_length:1618.0, max_img_length:1618
    """
    
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    model_type = ModelType.qwen_vl
        
    template_type = get_default_template_type(model_type)
    print(f'template_type: {template_type}')  # template_type: qwen

    model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

    # model = Swift.from_pretrained(model, inference_mode=True) #加载微调后的模型
    template = get_template(template_type, tokenizer)
    data_structure = ["perception", "prediction", "planning", "behavior"]
    
    count = {"que":[], "qa_length":[], "q_length":[], "a_length":[], "img_length":[]}
    for i in tqdm(range(len(list(data.keys())[:100]))):
        scene_id = list(data.keys())[i]
        scene_value = data[scene_id]
    # for scene_id, scene_value in data.items():
        # print(scene_id, type(scene_value), scene_value.keys(), type(scene_value["key_frames"]))
        for frame_id, value in scene_value["key_frames"].items():
            questions = []
            answers = []
            qas_list = []
            image_paths = value["image_paths"]
            # for key, v in image_paths.items():
            #     image_paths[key] = "Picture 1:<img>/home/ldl/pi_code/DriveLM/challenge/llama_adapter_v2_multimodal7b/data/nuscenes/samples/CAM_FRONT/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404.jpg</img>\n"
            QA = value["QA"]
            for name in data_structure:
                qas = QA[name]
                for qa in qas:
                    question = qa["Q"]
                    answer = qa["A"]
                    if "tag" in qa.keys():
                        tag = qa["tag"]
                    else:
                        tag = ""
                        pass
                    # print(question, answer, tag)
                    one_qa = question + ";" + answer
                    questions.append(question)
                    answers.append(answer)
                    qas_list.append(one_qa)
                    
                    length = len(qas_list)
                    start = (length-60) if length>60 else 0
                    query = ". ".join(qas_list[start:length])
                    qa_length = get_length(model, template, query)
                    query = ". ".join(questions[start:length])
                    q_length = get_length(model, template, query)
                    query = ". ".join(answers[start:length])
                    a_length = get_length(model, template, query)
            img_length = get_length(model, template, str(image_paths))
            # print(f"len(questions): {len(questions)}; length of query: {length}")
            count["que"].append(len(questions))
            count["qa_length"].append(qa_length)
            count["q_length"].append(q_length)
            count["a_length"].append(a_length)
            count["img_length"].append(img_length)
            # raise
            # lengths = []
            # for ques in questions:
            #     length = get_length(model, template, ques)
            #     lengths.append(length)
            # print(lengths)
            # raise
            
    
    queries = np.array(count["que"])
    qa_lengths = np.array(count["qa_length"])
    q_lengths = np.array(count["q_length"])
    a_lengths = np.array(count["a_length"])
    ave_query = np.average(queries)
    max_query = np.max(queries)
    print(f"length of queries: {len(queries)}")
    print(f"ave_query:{ave_query}, max_query:{max_query}")
    print(f"ave_qa_length:{np.average(qa_lengths)}, max_qa_length:{np.max(qa_lengths)}, np.percentile(a,95):{np.percentile(qa_lengths,95)}, np.percentile(a,99):{np.percentile(qa_lengths,99)}")
    print(f"ave_q_length:{np.average(q_lengths)}, max_q_length:{np.max(q_lengths)}")
    print(f"ave_a_length:{np.average(a_lengths)}, max_a_length:{np.max(a_lengths)}")
    print(f"ave_img_length:{np.average(count['img_length'])}, max_img_length:{np.max(count['img_length'])}")
    
    # print(queries)
    # print(lengths)
    # raise
    # ids = paragraph["conversations"][0]["id"]
    # value = paragraph["conversations"][0]["value"]
    # lines = value.split("\n")
    # last_line = lines[-1].split("/n")[-1]
        
    