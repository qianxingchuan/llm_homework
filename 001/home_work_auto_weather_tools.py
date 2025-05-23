from ast import mod
import json

from dashscope import api_key
import dashscope_client
import ollama_client
from api_key_constants import dashscope_secret

from function_call_processor import function_call_processor


# llm_client = dashscope_client.DashscopeClient(api_key=dashscope_secret.get("api_key"))
llm_client = ollama_client.OllamaClient(model="qwen2.5-coder:1.5b")


def run_conversation(query):
    messages = [{
        "role": "system",
        "content": """
        你是一个专业的天气助手，仅能处理与天气相关的问题，其他问题需拒绝回答。
        """
    }]
    messages.append({"role": "user", "content": query})

    # 得到第一次响应
    message = llm_client.get_response(messages)
    messages.append(message)
    print('message=', message)

    # Step 2, 判断用户是否要call function
    round_index = 1
    # 判断有没有 function_call 属性，如果没有，则不处理
    tool_info = function_call_processor.process_function_call(llm_client,message)
    while tool_info:
        if tool_info:
            messages.append(tool_info)
            print('messages=', messages)
            # Step 4, 代入function执行结果，再次调用模型
            round_index += 1
            message = llm_client.get_response(messages)
            messages.append(message)
            # 如果content 有<think></think>，则继续循环
            think_round = 0
            while message['content'].find('<think>') != -1:
                think_round += 1
                print("第 %d 轮有think,think %d round, %s" % (round_index , think_round , message['content']))
                messages.append({
                    "role": "tool",
                    "content":"我现在是一个自动化程序，查看上下文tool给出的信息，对于最终结果还缺少的部分，你需要再次调用function来获取。"
                })
                message = llm_client.get_response(messages)
                messages.append(message)
                continue;
            tool_info = function_call_processor.process_function_call(llm_client,message)
        else:
            print("第 %d 轮没有function_call" % round_index)
            break;
    # 最终结果
    return llm_client.fetch_content(message)


query_content = "杭州今天天气怎么样？"
result = run_conversation(query_content)
if result:
    print("最终结果:", result)
else:
    print("对话执行失败")
