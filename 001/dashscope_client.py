from typing import override
import dashscope
from llm_client import LLMClient

class DashscopeClient(LLMClient):
    """Dashscope LLM客户端实现"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.tools=[
            {
                'name': 'get_location_by_ip',
                'description': 'Get current city location.',
                'parameters': {
                }
            },
            {
                'name': 'get_location_id_by_city_name',
                'description': 'Get location id by city name.',
                'parameters': {
                    'type':'object',
                    "properties":{
                        "city":{
                            "type":"string",
                            "description":"The city name, e.g. San Francisco"
                        }
                    }
                }
            },
            {
                "name": "get_current_weather",
                "description": "获取当前城市的天气，必须按照准确日期和locationId进行查询",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "locationId": {"type": "string", "description": "城市Id,通过get_location_id_by_city_name获取到的值"},
                        "date": {"type": "string", "description": "日期（YYYY-MM-DD）"}
                    },
                    "required": ["locationId","date"]
                }
            },{
                "name":"today",
                "description":"获取今天的日期",
                "parameters":{
                }
            }
        ]
        dashscope.api_key = self.api_key

    @override
    def get_response(self, messages: list) -> object:
        try:
            response=dashscope.Generation.call(
                model='qwen-turbo',
                messages=messages,
                functions=self.tools,
                result_format='message'
            )
            return response.output.choices[0].message
        except Exception as e:
            print(f"API调用出错: {str(e)}")
            return None

    @override
    def should_process_function_call(self, message: object) -> bool:
        try:
            return hasattr(message, 'function_call') and message.function_call
        except Exception as e:
            print(f"<UNK> {message} <UNK>: {str(e)}")
            return False

    @override
    def llm_name(self) -> str:
        return LLMClient.LLM_DASHSCOPE
    
    @override
    def fetch_function_call_obj(self, message: object) -> dict:
        try:
            return message.function_call
        except Exception as e:
            print(f"<UNK> {message} <UNK>: {str(e)}")
            return None
    
    @override
    def fetch_content(self, message: object) -> str:
        try:
            return message.content
        except Exception as e:
            print(f"<UNK> {message} <UNK>: {str(e)}")
            return None