# 测试ollama
from openai import OpenAI
import os

# base_url ="https://dashscope.aliyuncs.com/compatible-mode/v1"
base_url ="http://localhost:11434/api"
# model="qwen-plus"
model="qwen3:0.6b"

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=base_url,
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
)
print(completion.model_dump_json())