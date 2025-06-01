from abc import ABC, abstractmethod
class LLMClient(ABC):
    """LLM客户端抽象接口，定义所有LLM客户端需要实现的核心方法"""

    # 定义两个常量
    LLM_DASHSCOPE = "dashscope"
    LLM_OLLAMA = "ollama"
    
    @abstractmethod
    def get_response(self, messages: list) -> object:
        """
        获取LLM的响应
        :param messages: 对话消息列表
        :return: LLM响应对象
        """
        pass

    @abstractmethod
    def should_process_function_call(self, message: object) -> bool:
        """
        判断是否需要处理function call
        :param message: LLM返回的消息对象
        :return: 是否需要处理
        """
        pass

    @abstractmethod
    def llm_name(self) -> str:
        """
        获取LLM的名称
        :return: LLM名称
        """
        pass

    @abstractmethod
    def fetch_function_call_obj(self, message: object) -> dict:
        """
        从LLM返回的消息对象中提取function call对象
        :param message: LLM返回的消息对象
        :return: function call对象
        """
        pass

    @abstractmethod
    def fetch_content(self, message: object) -> str:
        """
        从LLM返回的消息对象中提取content
        :param message: LLM返回的消息对象
        :return: content字符串
        """
        pass