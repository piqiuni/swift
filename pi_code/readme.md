# Readme
```bash
cd pi_code/swift/
conda activate swift
```

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

# TODO:
## 运行推理
目前`pi_code/infer_qwen_vl.py`调用的是原始模型`model_type = ModelType.qwen_vl`, 需要修改为加载微调后的checkpoints

参考：
[Qwen-VL 最佳实践](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/qwen-vl%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md)——给出基础的推理代码，无法解决问题

[LLM推理文档](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E6%8E%A8%E7%90%86%E6%96%87%E6%A1%A3.md#qwen-vl-chat)——参考qwen-7b的对话方式，修改推理代码


## 修改推理文件
指定推理使用测试集`~/pi_code/swift/pi_code/mini_trainning_llama.json`

## 修改输出文件格式
格式参考`swift/pi_code/pi_output.json`
需要保存id, 直接在`pi_code/infer_qwen_vl.py`里修改，读取每一条，推理后存储每一条

