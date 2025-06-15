from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_community.llms import Tongyi
from langchain.memory import ConversationBufferMemory
import re
import json
from typing import List, Union, Dict, Any
import os

# 设置通义千问API密钥
DASHSCOPE_API_KEY = 'sk-882e296067b744289acf27e6e20f3ec0'

# 自定义工具1：文本分析工具
class TextAnalysisTool:
    """文本分析工具，用于分析文本内容"""
    
    def __init__(self):
        self.name = "文本分析"
        self.description = "分析文本内容，提取字数、字符数和情感倾向"
    
    def run(self, text: str) -> str:
        """分析文本内容
        
        参数:
            text: 要分析的文本
        返回:
            分析结果
        """
        # 简单的文本分析示例
        word_count = len(text.split())
        char_count = len(text)
        
        # 简单的情感分析（示例）
        positive_words = ["好", "优秀", "喜欢", "快乐", "成功", "美好"]
        negative_words = ["差", "糟糕", "讨厌", "悲伤", "失败", "痛苦"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        sentiment = "积极" if positive_count > negative_count else "消极" if negative_count > positive_count else "中性"
        
        return f"文本分析结果:\n- 字数: {word_count}\n- 字符数: {char_count}\n- 情感倾向: {sentiment}"

# 自定义工具2：数据转换工具
class DataConversionTool:
    """数据转换工具，用于在不同格式之间转换数据"""
    
    def __init__(self):
        self.name = "数据转换"
        self.description = "在不同数据格式之间转换，如JSON、CSV等"
    
    def run(self, input_data: str, input_format: str, output_format: str) -> str:
        """转换数据格式
        
        参数:
            input_data: 输入数据
            input_format: 输入格式
            output_format: 输出格式
        返回:
            转换后的数据
        """
        try:
            if input_format.lower() == "json" and output_format.lower() == "csv":
                # JSON到CSV的转换示例
                data = json.loads(input_data)
                if isinstance(data, list):
                    if not data:
                        return "空数据"
                    
                    # 获取所有可能的列
                    headers = set()
                    for item in data:
                        headers.update(item.keys())
                    headers = list(headers)
                    
                    # 创建CSV
                    csv = ",".join(headers) + "\n"
                    for item in data:
                        row = [str(item.get(header, "")) for header in headers]
                        csv += ",".join(row) + "\n"
                    
                    return csv
                else:
                    return "输入数据必须是JSON数组"
            
            elif input_format.lower() == "csv" and output_format.lower() == "json":
                # CSV到JSON的转换示例
                lines = input_data.strip().split("\n")
                if len(lines) < 2:
                    return "CSV数据至少需要标题行和数据行"
                
                headers = lines[0].split(",")
                result = []
                
                for line in lines[1:]:
                    values = line.split(",")
                    if len(values) != len(headers):
                        continue
                    
                    item = {}
                    for i, header in enumerate(headers):
                        item[header] = values[i]
                    result.append(item)
                
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"不支持的转换: {input_format} -> {output_format}"
        
        except Exception as e:
            return f"转换失败: {str(e)}"

# 自定义工具3：文本处理工具
class TextProcessingTool:
    """文本处理工具，用于处理文本内容"""
    
    def __init__(self):
        self.name = "文本处理"
        self.description = "处理文本内容，如查找、替换、统计等"
    
    def run(self, operation: str, content: str, **kwargs) -> str:
        """处理文本内容
        
        参数:
            operation: 操作类型
            content: 文本内容
            **kwargs: 其他参数
        返回:
            处理结果
        """
        if operation == "count_lines":
            return f"文本共有 {len(content.splitlines())} 行"
        
        elif operation == "find_text":
            search_text = kwargs.get("search_text", "")
            if not search_text:
                return "请提供要查找的文本"
            
            lines = content.splitlines()
            matches = []
            
            for i, line in enumerate(lines):
                if search_text in line:
                    matches.append(f"第 {i+1} 行: {line}")
            
            if matches:
                return f"找到 {len(matches)} 处匹配:\n" + "\n".join(matches)
            else:
                return f"未找到文本 '{search_text}'"
        
        elif operation == "replace_text":
            old_text = kwargs.get("old_text", "")
            new_text = kwargs.get("new_text", "")
            
            if not old_text:
                return "请提供要替换的文本"
            
            new_content = content.replace(old_text, new_text)
            count = content.count(old_text)
            
            return f"替换完成，共替换 {count} 处。\n新内容:\n{new_content}"
        
        else:
            return f"不支持的操作: {operation}"

# 创建工具链
def create_tool_chain():
    """创建工具链"""
    # 创建工具
    text_analysis = TextAnalysisTool()
    data_conversion = DataConversionTool()
    text_processing = TextProcessingTool()
    
    # 组合工具
    tools = [
        Tool(
            name=text_analysis.name,
            func=text_analysis.run,
            description="分析文本内容，提取字数、字符数和情感倾向"
        ),
        Tool(
            name=data_conversion.name,
            func=data_conversion.run,
            description="在不同数据格式之间转换，如JSON、CSV等"
        ),
        Tool(
            name=text_processing.name,
            func=text_processing.run,
            description="处理文本内容，如查找、替换、统计等"
        )
    ]
    
    # 初始化语言模型
    llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)
    
    # 创建提示模板
    prompt = PromptTemplate.from_template(
        """你是一个有用的AI助手，可以使用以下工具:
{tools}
可用工具名称: {tool_names}

使用以下格式:
问题: 你需要回答的问题
思考: 你应该始终思考要做什么
行动: 要使用的工具名称，必须是 [{tool_names}] 中的一个
行动输入: 工具的输入
观察: 工具的结果
... (这个思考/行动/行动输入/观察可以重复 N 次)
思考: 我现在已经有了最终答案
回答: 对原始问题的最终回答

开始!
问题: {input}
思考: {agent_scratchpad}"""
    )
    
    # 创建Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # 创建Agent执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        verbose=True,
        handle_parsing_errors=False  # 关闭自动重试, True会严格检查重试
    )
    
    return agent_executor

# 示例：使用工具链处理任务
def process_task(task_description):
    """
    使用工具链处理任务
    
    参数:
        task_description: 任务描述
    返回:
        处理结果
    """
    try:
        agent_executor = create_tool_chain()
        response = agent_executor.invoke({"input": task_description})
        return response["output"]  # 从返回的字典中提取输出
    except Exception as e:
        return f"处理任务时出错: {str(e)}"

# 示例用法
if __name__ == "__main__":
    # 示例1: 文本分析与处理
    task1 = "分析以下文本的情感倾向，并统计其中的行数：'这个产品非常好用，我很喜欢它的设计，使用体验非常棒！\n价格也很合理，推荐大家购买。\n客服态度也很好，解答问题很及时。'"
    print("任务1:", task1)
    print("结果:", process_task(task1))
    
    # 示例2: 数据格式转换
    task2 = "将以下CSV数据转换为JSON格式：'name,age,comment\n张三,25,这个产品很好\n李四,30,服务态度差\n王五,28,性价比高'"
    print("\n任务2:", task2)
    print("结果:", process_task(task2))