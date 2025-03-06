import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# 模型名称
model_name = "Qwen/QwQ-32B-AWQ"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 获取 GPU 信息
def get_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)  # 获取第一个 GPU 的型号
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # 总显存（GB）
        used_memory = torch.cuda.memory_allocated(0) / 1024**3  # 已用显存（GB）
        free_memory = total_memory - used_memory  # 剩余显存（GB）
        return f"GPU 型号: {gpu_name}\n总显存: {total_memory:.2f} GB\n已用显存: {used_memory:.2f} GB\n剩余显存: {free_memory:.2f} GB"
    else:
        return "未检测到 GPU，使用 CPU"

# 定义流式文本生成函数并计算时间、速度和 GPU 信息
def generate_response(prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    
    # 生成对话格式文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 进行 token 化并移动到模型的设备
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 记录生成开始时间
    start_time = time.time()
    
    # 生成文本
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    
    # 记录生成结束时间
    end_time = time.time()
    
    # 计算生成时间（秒）
    generation_time = end_time - start_time
    
    # 提取生成的 token，去除输入部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # 解码生成结果
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 计算生成的 token 数量
    num_generated_tokens = len(generated_ids[0])
    
    # 计算生成速度（tokens/秒）
    if generation_time > 0:  # 避免除以零
        token_speed = num_generated_tokens / generation_time
    else:
        token_speed = 0
    
    # 获取 GPU 信息
    gpu_info = get_gpu_info()
    
    # 返回响应、生成时间、速度和 GPU 信息
    return f"{response}\n\n生成时间: {generation_time:.2f} 秒\n生成速度: {token_speed:.2f} tokens/秒\n\n{gpu_info}"

# 创建 Gradio 界面
interface = gr.Interface(
    fn=generate_response,  
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
        gr.Slider(minimum=1, maximum=32768, step=1, value=512, label="Max new tokens")
    ],
    outputs=gr.Textbox(label="Response, Time, Speed, and GPU Info"),  # 更新输出标签
    title="Qwen/QwQ-32B-AWQ Chat Interface",
    description="Enter a prompt and get a response from the Qwen/QwQ-32B-AWQ model with generation time, speed, and GPU info."
)

# 运行 Gradio
interface.launch(share=True)
