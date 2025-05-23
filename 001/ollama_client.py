from typing import override
import ollama
from llm_client import LLMClient  # 导入抽象基类

class OllamaClient(LLMClient):
    """Ollama LLM客户端实现（需Ollama服务已启动）"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:0.6b"):
        """
        :param base_url: Ollama服务地址（默认本地服务）
        :param model: 使用的模型名称（默认qwen3:0.6b）
        """
        self.client = ollama.Client(
            host=base_url,  # 确保host参数正确设置为base_url
        )
        self.base_url = base_url
        self.model = model
        self.tools= [
            {
                "type": "function",
                "function": {
                    "name": "weather_tools.get_location_by_ip",
                    "description": "Get current city location.",
                    "parameters": {
                        "type": "object",
                        "properties": {},  # No parameters required
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "weather_tools.get_location_id_by_city_name",
                    "description": "Get location id by city name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city_name": {
                                "type": "string",
                                "description": "城市名称，会返回城市Id"
                            }
                        },
                        "required": ["city_name"]  # city is required to get location ID
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "weather_tools.get_current_weather",
                    "description": "获取当前城市的天气，必须按照准确日期和城市Id进行查询",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "locationId": {
                                "type": "string",
                                "description": "城市Id,需要依赖get_location_id_by_city_name获取到的值"
                            },
                            "date": {
                                "type": "string",
                                "description": "日期（YYYY-MM-DD），需要依赖传入正确的日期，如果是今天，需要依赖today函数获取到的值"
                            }
                        },
                        "required": ["locationId", "date"]  # Both fields are required
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "weather_tools.today",
                    "description": "获取今天的日期，格式为YYYY-MM-DD",
                    "parameters": {
                        "type": "object",
                        "properties": {},  # No parameters required
                        "required": []
                    }
                }
            }
        ]

    @override
    def get_response(self, messages: list) -> object:
        """
        调用Ollama API获取响应（支持工具调用引导）
        :param messages: 对话消息列表（需包含引导工具调用的system prompt）
        :param functions: 工具函数描述列表（用于构造system prompt）
        """
        try:
            # 构造包含工具描述的system prompt（替代直接传functions参数）
            response =self.client.chat(self.model, messages=messages,tools=self.tools,stream=False)
            # 返回这一轮的message，如果没有，抛异常
            if response and "message" in response:
                return response["message"]
            else:
                raise Exception("Ollama API返回结果格式错误")
        except Exception as e:
            print(f"Ollama API调用出错: {str(e)}")
            return None

    @override
    def should_process_function_call(self, message: object) -> bool:
        """
        判断是否需要处理工具调用（通过文本匹配识别工具调用指令）
        :param message: 响应中的message对象（含content字段）
        :return: 是否包含工具调用标记
        """
        if not message or "tool_calls" not in message:
            return False
        # 简单判断是否包含工具调用标记
        return any("function" in call for call in message["tool_calls"])

    @override
    def llm_name(self) -> str:
        return LLMClient.LLM_OLLAMA

    @override
    def fetch_function_call_obj(self, message: object) -> dict:
        """
        从LLM返回的消息对象中提取function call对象
        :param message: LLM返回的消息对象
        :return: function call对象
        """
        if not self.should_process_function_call(message):
            return None
        # 假设tool_calls字段中第一个调用是有效的
        return message["tool_calls"][0]["function"]
    
    @override
    def fetch_content(self, message: object) -> str:
        """
        从LLM返回的消息对象中提取content
        :param message: LLM返回的消息对象
        :return: content字符串
        """
        return message.get("content", "")