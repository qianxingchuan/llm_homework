import weather_tools
from llm_client import LLMClient
import json
class function_call_processor:
    """Function call处理工具类"""
    
    @staticmethod
    def process_function_call(llm_client,message: object) -> dict:
        """
        处理function call
        :param message: LLM返回的消息对象
        :return: function call结果信息
        """
        if not llm_client.should_process_function_call(message):
            print("未检测到function call")
            return None
            
        function_call = llm_client.fetch_function_call_obj(message)
        tool_name = function_call['name']
        # 去除前缀，只保留函数名部分
        tool_name = tool_name.replace('weather_tools.', '')  
        args = function_call['arguments']
        # 如果args是字符串，则转换为字典
        if isinstance(args, str):
            args = json.loads(args)
        arguments = args
        tool_response = function_call_processor.call_function(tool_name, **arguments)
        
        # 确保响应为字符串格式
        if not isinstance(tool_response, str):
            tool_response = json.dumps(tool_response)
        
        # 根据不同的LLM返回不同的格式
        if llm_client.llm_name() == LLMClient.LLM_OLLAMA:
            return {"role": "tool", "name": tool_name, "content": tool_response}
        return {"role": "function", "name": tool_name, "content": tool_response}
        
    @staticmethod
    def call_function(function_name: str, **kwargs) -> any:
        """
        调用具体工具函数
        :param function_name: 工具函数名
        :param kwargs: 参数
        :return: 函数执行结果
        """
        method = getattr(weather_tools, function_name)
        if not callable(method):
            print(f"函数 {function_name} 不存在或不可调用")
            return None
            
        try:
            import inspect
            sig = inspect.signature(method)
            sig.bind(**kwargs)  # 验证参数有效性
            return method(**kwargs)
        except TypeError as e:
            print(f"调用函数 {function_name} 时参数出错: {str(e)}")
            return None