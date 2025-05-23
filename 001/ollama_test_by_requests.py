import requests
import json
from ollama_client import OllamaClient  # 导入OllamaClient类

url="http://localhost:11434/api/chat"
tools = OllamaClient().tools
# 定义请求头
headers = {
    "Content-Type": "application/json"
}

messages =[
        {"role": "system", 
        "content": 
        """
        你是一个天气助手，你需要根据用户的输入，返回相应的天气信息。
        你需要拆解用户的输入，通过tool的处理来获得最终的信息，不允许通过自己的知识库直接返回数据
        """}
    ]

# 定义请求体
data = {
    "model":"qwen3:4b",
    "messages": messages,
    "stream": False,
    "options": {
        "temperature": 0
    },
    "tools": tools
}

def check_weather():
    user_query="今天杭州的天气怎么样?"
    messages.append({"role": "user", "content": user_query})
    data["messages"]=messages
    response = requests.post(url,data=json.dumps(data),headers=headers)
    # 如果有message字段
    round_index=0
    while response and "message" in response.json():
        round_index+=1
        message=response.json()["message"]
        messages.append(message)
        # 如果有tool_calls字段，取第一个调用，处理后，给到上下文
        if "tool_calls" in message:
            tool_calls=message["tool_calls"]
            # 遍历执行tool_calls
            tool_info_list = []
            for tool_call in tool_calls:
                tool_info = process_toll_call(tool_call['function'],round_index)
                tool_info_list.append(tool_info)
            
            # tool_info_list加到messages
            messages.extend(tool_info_list)
        else:
            print(f"返回结果是{message['content']}")
            break
        # 重新请求
        data["messages"]=messages
        response = requests.post(url,data=json.dumps(data),headers=headers)
    return response.json()

def process_toll_call(tool_call,round_index):
    print(f"the {round_index} round,调用的tool是{tool_call}")
    tool_name = tool_call["name"]
    tool_info = {"role":"user","content":""}
    if tool_name == "weather_tools.today" or tool_name == "today":
        print(f"调用get_today,参数是{tool_call["arguments"]}")
        tool_info["content"]=tool_name + "的值是：2025-05-23"
    if tool_name == "weather_tools.get_location_id_by_city_name" or tool_name == "get_location_id_by_city_name":
        print(f"调用get_location_id_by_city_name,参数是{tool_call["arguments"]}")
        tool_info["content"]=tool_name + "的值是：hangzhou.xingchuan.qxc"
    if tool_name == "weather_tools.get_current_weather" or tool_name == "get_current_weather":
        print(f"调用get_current_weather,参数是{tool_call["arguments"]}")
        # 判断arguments必须包含locationId和date
        if "locationId" not in tool_call["arguments"] or "date" not in tool_call["arguments"]:
            raise Exception("参数错误")
        # 判断locationId必须是hangzhou.xingchuan.qxc
        if tool_call["arguments"]["locationId"] != "hangzhou.xingchuan.qxc":
            raise Exception("城市ID参数错误")
        # 判断date必须是2025-05-23
        if tool_call["arguments"]["date"]!= "2025-05-23":
            raise Exception("日期参数错误")
        tool_info["content"]=tool_name + "的值是："+"""
        {
            "code": 200,
            "message": "success",
            "data": {
                "locationId": "hangzhou.code",
                "date": "2025-05-23",
                "weather": "晴",
                "temperature": "25℃",
                "wind": "无持续风向 3级",
                "humidity": "50%",
                "airQuality": "优",
            }
        }
        """
    return tool_info

# 打印响应内容
response = check_weather()
result_content = response["message"]["content"]
print(result_content)


