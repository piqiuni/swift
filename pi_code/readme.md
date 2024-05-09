# Readme
```bash
cd pi_code/swift/
conda activate swift
```

# Reinstall after pull
`pip install -e '.[llm]'`

https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation?tab=readme-ov-file

## Fine-tuning the model
运行
```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen-vl \
    --custom_train_dataset_path ~/pi_code/swift/pi_code/mini_trainning_llama.json \
    --output_dir output \
```

或通过python运行
```bash
cd ~/pi_code/swift
python pi_code/fine_tune_qwen_vl.py
```

输出文件目录：

`output/qwen-vl`

报错处理：
```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```


## Infer
命令行运行——存在问题，无法调用测试集，无法保存为期望格式
```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true\
```

Python运行
```bash
cd ~/pi_code/swift
python pi_code/infer_qwen_vl.py
```

## Eval
```bash
conda activate llama_adapter_v2
cd ~/pi_code/DriveLM/challenge
python evaluation.py --root_path1 ../../swift/pi_code/output/output_0425.json --root_path2 ./mini_test_eval.json

python evaluation.py --root_path1 ../../swift/pi_code/output/output_0425.json --root_path2 ./test_eval.json
```
修改 `--root_path1`


参考：
[Qwen-VL 最佳实践](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/qwen-vl%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md)——给出基础的推理代码，无法解决问题

[命令行参数](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.md)

[LLM推理文档](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E6%8E%A8%E7%90%86%E6%96%87%E6%A1%A3.md#qwen-vl-chat)——参考qwen-7b的对话方式，修改推理代码

# TODO
Ref: 文件格式参考
```
/home/ldl/pi_code/DriveLM/challenge/pi_test/data_structure/v1_1_test.json
/home/ldl/pi_code/DriveLM/challenge/pi_test/data_structure/v1_1_train.json
```


1. 数据集处理  
   修改训练数据，加入**60(n可调)帧滑动窗口**，将历史数据输入训练，同时**修改图片路径**  
   修改`/home/ldl/pi_code/DriveLM/challenge/pi_test/pi_transfer_data_history.py`文件实现训练数据转换
   1. 修改图片路径————减小图片路径输入length  
        原始`data/nuscenes/samples`
        新路径`<img>samples/CAM_FRONT/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404.jpg</img>`  
   2. 修改  
        具体要求参见`pi_transfer_data_history.py`前部分注释
        目的，引入60帧滑动窗口，将历史数据输入训练，仅考虑每个frame关键帧中的历史问答数据
        ```
        
        ```
   3. 增加场景描述 
        根据每个场景的'scene_description', 将其加入到每个场景的QA中
        ```
        Q: "Describe the scene."
        A: scene_description
        ```


2. 模型微调处理——暂不修改，有Bug
   1. 修改模型为`ModelType.internlm_xcomposer2_7b_chat`——更强的上下文理解能力
   2. 模型`max_length=4096,` maybe longer？
   
3. 模型推理处理
   1. 上下文信息读取，参考`/home/ldl/pi_code/swift/pi_code/infer_qwen_vl.py`
        ```
        elif model_type == ModelType.qwen_vl_chat:
            his_length = len(history)
            start_index = max(0, his_length - max_his_length)
            history = history[start_index:]
            response, _ = inference(model, template, value, history)
            qa = [last_line, response]
            history.append(qa)
        ```
   2. 输出保存只保存当前问题
   
















# 训练文件格式
原始训练文件 `/home/ldl/pi_code/data/DriveLM/v1_1_train_nus.json`
```
{
    "f0f120e4d4b0441da90ec53b16ee169d": {
        "scene_description": "The ego vehicle proceeds through the intersection, continuing along the current roadway.",
        "key_frames": {
            "4a0798f849ca477ab18009c3a20b7df2": {
                "key_object_infos": {
                    "<c1,CAM_BACK,1088.3,497.5>": {
                        "Category": "Vehicle",
                        "Status": "Moving",
                        "Visual_description": "Brown SUV.",
                        "2d_bbox": [
                            966.6,
                            403.3,
                            1224.1,
                            591.7
                        ]
                    },
                    "<c2,CAM_BACK,864.2,468.3>": {
                        "Category": "Vehicle",
                        "Status": "Moving",
                        "Visual_description": "Black sedan.",
                        "2d_bbox": [
                            816.7,
                            431.6,
                            917.2,
                            505.0
                        ]
                    },
                    "<c3,CAM_FRONT,1043.2,82.2>": {
                        "Category": "Traffic element",
                        "Status": null,
                        "Visual_description": "Green light.",
                        "2d_bbox": [
                            676.4,
                            0.0,
                            1452.6,
                            171.5
                        ]
                    }
                },
                "QA": {
                    "perception": [
                        {
                            "Q": "What are objects to the front right of the ego car?",
                            "A": "There are many barriers and one construction vehicle to the front right of the ego car.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What are objects to the front left of the ego car?",
                            "A": "There is one truck and one barrier to the front left of the ego car.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the status of the truck that is to the front left of the ego car?",
                            "A": "One truck is moving.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What are objects to the back left of the ego car?",
                            "A": "There are two barriers, many trucks, two trailers, and one car to the back left of the ego car.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the status of the trucks that are to the back left of the ego car?",
                            "A": "Many trucks are parked.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the status of the construction vehicle that is to the front right of the ego car?",
                            "A": "The construction vehicle to the front right of the ego car is parked.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What are objects to the front of the ego car?",
                            "A": "There are many obstacles in front of the ego car.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the status of the trailers that are to the back left of the ego car?",
                            "A": "Two trailers are parked.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What are objects to the back of the ego car?",
                            "A": "There are two cars, one truck, and one barrier to the back of the ego car.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the status of the cars that are to the back of the ego car?",
                            "A": "Two cars are moving.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the status of the car that is to the back left of the ego car?",
                            "A": "One car is moving.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the status of the truck that is to the back of the ego car?",
                            "A": "One truck is parked.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What are objects to the back right of the ego car?",
                            "A": "There are two trailers to the back right of the ego car.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the status of the trailers that are to the back right of the ego car?",
                            "A": "One of the trailers is parked, and one is moving.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Are there barriers to the front right of the ego car?",
                            "A": "Yes.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Are there standing pedestrians to the front right of the ego car?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Are there parked trailers to the front right of the ego car?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Are there barriers to the front of the ego car?",
                            "A": "Yes.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Are there parked trucks to the back left of the ego car?",
                            "A": "Yes.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Are there parked trailers to the back right of the ego car?",
                            "A": "Yes.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Are there barriers to the front left of the ego car?",
                            "A": "Yes.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Are there parked construction vehicles to the front right of the ego car?",
                            "A": "Yes.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Please describe the current scene.",
                            "A": "There are two moving cars behind the ego car and two barriers in front of it.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the relative positioning of the important objects in the current scene?",
                            "A": "<c2,CAM_BACK,864.2,468.3> is at the back of <c1,CAM_BACK,1088.3,497.5>.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Is there any traffic element in the front view?",
                            "A": "Yes, there are some traffic elements in the front view.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Identify all the traffic elements in the front view, categorize them, determine their status, and predict the bounding box around each one. The output should be a list formatted as (c, s, x1, y1, x2, y2), where c represents the category, s denotes the status, and x1, y1, x2, y2 are the offsets of the top-left and bottom-right corners of the box relative to the center point.",
                            "A": "There are two traffic elements in the front view. The information of these traffic elements is [(traffic light, green, 674.86, 0.14, 723.33, 109.18), (traffic light, green, 1018.98, 7.19, 1071.77, 125.6)].",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Which lanes are each important object on in the scene?",
                            "A": "<c1,CAM_BACK,1088.3,497.5> is in the left lane, and <c2,CAM_BACK,864.2,468.3> is in the ego lane.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.",
                            "A": "There is a brown SUV to the back of the ego vehicle, a black sedan to the back of the ego vehicle, and a green light to the front of the ego vehicle. The IDs of these objects are <c1,CAM_BACK,1088.3,497.5>, <c2,CAM_BACK,864.2,468.3>, and <c3,CAM_FRONT,1043.2,82.2>.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the observed status of object <c1,CAM_BACK,1088.3,497.5>?",
                            "A": "Moving.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the moving status of object <c1,CAM_BACK,1088.3,497.5>?",
                            "A": "Turn left.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the observed status of object <c2,CAM_BACK,864.2,468.3>?",
                            "A": "Moving.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the moving status of object <c2,CAM_BACK,864.2,468.3>?",
                            "A": "Going ahead.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the visual description of <c1,CAM_BACK,1088.3,497.5>?",
                            "A": "Brown SUV.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the visual description of <c2,CAM_BACK,864.2,468.3>?",
                            "A": "Black sedan.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the visual description of <c3,CAM_FRONT,1043.2,82.2>?",
                            "A": "Green light.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        }
                    ],
                    "prediction": [
                        {
                            "Q": "Is <c1,CAM_BACK,1088.3,497.5> a traffic sign or a road barrier?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Which object is most likely to be occluded by <c1,CAM_BACK,1088.3,497.5>? Would this object affect the ego vehicle? Based on this object, what action of the ego vehicle is dangerous?",
                            "A": "None, no, none.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Would <c1,CAM_BACK,1088.3,497.5> be in the moving direction of the ego vehicle?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the future state of <c1,CAM_BACK,1088.3,497.5>?",
                            "A": "Turn left.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Is <c2,CAM_BACK,864.2,468.3> a traffic sign or a road barrier?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Which object is most likely to be occluded by <c2,CAM_BACK,864.2,468.3>? Would this object affect the ego vehicle? Based on this object, what action of the ego vehicle is dangerous?",
                            "A": "None, no, none.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Would <c2,CAM_BACK,864.2,468.3> be in the moving direction of the ego vehicle?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the future state of <c2,CAM_BACK,864.2,468.3>?",
                            "A": "Turn left.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Is <c3,CAM_FRONT,1043.2,82.2> a traffic sign or a road barrier?",
                            "A": "Yes.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Would <c3,CAM_FRONT,1043.2,82.2> be in the moving direction of the ego vehicle?",
                            "A": "Yes.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What does <c3,CAM_FRONT,1043.2,82.2> mean?",
                            "A": "Please proceed.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "In this scenario, what object is most likely to consider <c3,CAM_FRONT,1043.2,82.2>?",
                            "A": "The ego vehicle.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What object should the ego vehicle notice first when the ego vehicle is getting to the next possible location? What is the state of the object that is first noticed by the ego vehicle and what action should the ego vehicle take? What object should the ego vehicle notice second when the ego vehicle is getting to the next possible location? What is the state of the object perceived by the ego vehicle as second and what action should the ego vehicle take? What object should the ego vehicle notice third? What is the state of the object perceived by the ego vehicle as third and what action should the ego vehicle take?",
                            "A": "Firstly notice that <c3,CAM_FRONT,1043.2,82.2>. The object is a traffic sign, so the ego vehicle should keep going ahead at the same speed. Secondly notice that <c1,CAM_BACK,1088.3,497.5>. The object is turning left, so the ego vehicle should keep going ahead at the same speed. Thirdly notice that <c2,CAM_BACK,864.2,468.3>. The object is going ahead, so the ego vehicle should keep going ahead at the same speed.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Are <c1,CAM_BACK,1088.3,497.5> and <c2,CAM_BACK,864.2,468.3> traffic signs?",
                            "A": "Neither is a traffic sign.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Will <c1,CAM_BACK,1088.3,497.5> be in the moving direction of <c2,CAM_BACK,864.2,468.3>?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Will <c1,CAM_BACK,1088.3,497.5> change its motion state based on <c2,CAM_BACK,864.2,468.3>?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Based on the observations of <c2,CAM_BACK,864.2,468.3>, what are possible actions to be taken by <c1,CAM_BACK,1088.3,497.5>? What is the reason?",
                            "A": "The action is to turn left, the reason is there is no safety issue.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What object would consider <c1,CAM_BACK,1088.3,497.5> to be most relevant to its decision?",
                            "A": "<c2,CAM_BACK,864.2,468.3>.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What object would consider <c2,CAM_BACK,864.2,468.3> to be most relevant to its decision?",
                            "A": "None.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Are <c1,CAM_BACK,1088.3,497.5> and <c3,CAM_FRONT,1043.2,82.2> traffic signs?",
                            "A": "Only one of the boxes is a traffic sign.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Will <c3,CAM_FRONT,1043.2,82.2> be in the moving direction of <c1,CAM_BACK,1088.3,497.5>?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Would <c1,CAM_BACK,1088.3,497.5> take <c3,CAM_FRONT,1043.2,82.2> into account?",
                            "A": "Yes.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What object would consider <c1,CAM_BACK,1088.3,497.5> to be most relevant to its decision?",
                            "A": "<c2,CAM_BACK,864.2,468.3>.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What object would consider <c3,CAM_FRONT,1043.2,82.2> to be most relevant to its decision?",
                            "A": "The ego vehicle.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Except for the ego vehicle, what object would consider <c3,CAM_FRONT,1043.2,82.2> to be most relevant to its decision?",
                            "A": "<c1,CAM_BACK,1088.3,497.5>.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What kind of traffic sign is <c3,CAM_FRONT,1043.2,82.2>?",
                            "A": "Traffic light.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        }
                    ],
                    "planning": [
                        {
                            "Q": "Is <c1,CAM_BACK,1088.3,497.5> an object that the ego vehicle should consider in the current scene?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What actions could the ego vehicle take based on <c1,CAM_BACK,1088.3,497.5>? Why take this action and what's the probability?",
                            "A": "The action is to keep going at the same speed. The reason is to follow the traffic rules, which has a high probability.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the probability of colliding with <c1,CAM_BACK,1088.3,497.5> after the ego vehicle goes straight and keeps the same speed?",
                            "A": "Low.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What actions taken by the ego vehicle can lead to a collision with <c1,CAM_BACK,1088.3,497.5>?",
                            "A": "No such action will lead to a collision.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Is <c2,CAM_BACK,864.2,468.3> an object that the ego vehicle should consider in the current scene?",
                            "A": "No.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What actions could the ego vehicle take based on <c2,CAM_BACK,864.2,468.3>? Why take this action and what's the probability?",
                            "A": "The action is to keep going at the same speed. The reason is to follow the traffic rules, which has a high probability.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the probability of colliding with <c2,CAM_BACK,864.2,468.3> after the ego vehicle accelerates and goes straight?",
                            "A": "Low.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What actions taken by the ego vehicle can lead to a collision with <c2,CAM_BACK,864.2,468.3>?",
                            "A": "Back up.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Is it necessary for the ego vehicle to take <c3,CAM_FRONT,1043.2,82.2> into account?",
                            "A": "Yes.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Based on <c3,CAM_FRONT,1043.2,82.2> in this scene, what is the most possible action of the ego vehicle?",
                            "A": "Keep going at the same speed.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the traffic signal that the ego vehicle should pay attention to?",
                            "A": "Green light.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the target action of the ego vehicle?",
                            "A": "Go straight.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "In this scenario, what are safe actions to take for the ego vehicle?",
                            "A": "Keep going at the same speed, decelerate gradually without braking.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "In this scenario, what are dangerous actions to take for the ego vehicle?",
                            "A": "Back up, turn right.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What is the priority of the objects that the ego vehicle should consider?(in descending order)",
                            "A": "<c3,CAM_FRONT,1043.2,82.2>, <c1,CAM_BACK,1088.3,497.5>, <c2,CAM_BACK,864.2,468.3>.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "Based on the observation of <c3,CAM_FRONT,1043.2,82.2>, what actions may <c1,CAM_BACK,1088.3,497.5> take?",
                            "A": "Turn left.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        },
                        {
                            "Q": "What will affect driving judgment in this scene?",
                            "A": "Water droplets on the glass will affect visibility.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        }
                    ],
                    "behavior": [
                        {
                            "Q": "Predict the behavior of the ego vehicle.",
                            "A": "The ego vehicle is going straight. The ego vehicle is driving fast.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null
                        }
                    ]
                },
                "image_paths": {
                    "CAM_FRONT": "../nuscenes/samples/CAM_FRONT/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404.jpg",
                    "CAM_FRONT_LEFT": "../nuscenes/samples/CAM_FRONT_LEFT/n008-2018-09-18-13-10-39-0400__CAM_FRONT_LEFT__1537291010604799.jpg",
                    "CAM_FRONT_RIGHT": "../nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-09-18-13-10-39-0400__CAM_FRONT_RIGHT__1537291010620482.jpg",
                    "CAM_BACK": "../nuscenes/samples/CAM_BACK/n008-2018-09-18-13-10-39-0400__CAM_BACK__1537291010637558.jpg",
                    "CAM_BACK_LEFT": "../nuscenes/samples/CAM_BACK_LEFT/n008-2018-09-18-13-10-39-0400__CAM_BACK_LEFT__1537291010647405.jpg",
                    "CAM_BACK_RIGHT": "../nuscenes/samples/CAM_BACK_RIGHT/n008-2018-09-18-13-10-39-0400__CAM_BACK_RIGHT__1537291010628113.jpg"
                }
            },

```