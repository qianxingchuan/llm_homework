import dashscope
from llm_client import LLMClient
from inventory_api import InventoryAPI

class DashscopeClient(LLMClient):
    """Dashscope LLM客户端实现"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api = InventoryAPI();
        self.tools=self.api.get_tools()
        dashscope.api_key = self.api_key

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

    def should_process_function_call(self, message: object) -> bool:
        try:
            return hasattr(message, 'function_call') and message.function_call
        except Exception as e:
            print(f"<UNK> {message} <UNK>: {str(e)}")
            return False

    def llm_name(self) -> str:
        return LLMClient.LLM_DASHSCOPE
    
    def fetch_function_call_obj(self, message: object) -> dict:
        try:
            return message.function_call
        except Exception as e:
            print(f"<UNK> {message} <UNK>: {str(e)}")
            return None
    
    def fetch_content(self, message: object) -> str:
        try:
            return message.content
        except Exception as e:
            print(f"<UNK> {message} <UNK>: {str(e)}")
            return None