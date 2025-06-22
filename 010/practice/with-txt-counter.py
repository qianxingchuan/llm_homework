"""基于 Assistant 实现的桌面 TXT 文件管理助手

这个模块提供了一个智能文件助手，可以：
1. 通过 HTTP 方式连接到 MCP 服务器
2. 支持多种交互方式（GUI、TUI、测试模式）
3. 支持文件统计、列表查看、内容读取等功能
"""

import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

def init_agent_service():
    """初始化桌面 TXT 文件管理助手服务
    
    配置说明：
    - 使用 qwen-max 作为底层语言模型
    - 设置系统角色为文件管理助手
    - 通过 HTTP 连接到 txt_counter MCP 服务器
    
    Returns:
        Assistant: 配置好的文件管理助手实例
    """
    # LLM 模型配置
    llm_cfg = {
        'model': 'qwen-max',
        'timeout': 30,  # 设置模型调用超时时间
        'retry_count': 3,  # 设置重试次数
    }
    
    # 系统角色设定
    system = ('你扮演一个桌面文件管理助手，你具有查看、统计、读取桌面上 TXT 文件的能力。'
             '你可以帮助用户统计桌面上的 TXT 文件数量，列出所有 TXT 文件，以及读取指定文件的内容。'
             '你应该充分利用提供的文件管理工具来提供专业的文件操作服务。'
             '当用户询问桌面文件相关问题时，请主动使用相应的工具来获取准确信息。')
    
    # MCP 工具配置 - 通过 HTTP 连接到已启动的 MCP 服务器
    tools = [{
        "mcpServers": {
            "txt-counter": {
                "type":"http",
                "url":"http://localhost:8080/sse",
                "headers":{
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "x-mcp-proxy-auth":"Bearer 1e44c421c38d273a4cba0e91588fad5b41615bf27198c5c7ddf898c812f1e15e"
                },
                "timeout":30
            }
        }
    }]
    
    try:
        # 创建助手实例
        bot = Assistant(
            llm=llm_cfg,
            name='桌面文件助手',
            description='桌面 TXT 文件管理与查询',
            system_message=system,
            function_list=tools,
        )
        print("桌面文件助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise


def test(query='帮我统计一下桌面上有多少个 TXT 文件', file: Optional[str] = None):
    """测试模式
    
    用于快速测试单个查询
    
    Args:
        query: 查询语句，默认为统计 TXT 文件数量
        file: 可选的输入文件路径
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 构建对话消息
        messages = []

        # 根据是否有文件输入构建不同的消息格式
        if not file:
            messages.append({'role': 'user', 'content': query})
        else:
            messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

        print("正在处理您的请求...")
        # 运行助手并打印响应
        for response in bot.run(messages):
            print('bot response:', response)
    except Exception as e:
        print(f"处理请求时出错: {str(e)}")


def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('user question: ')
                # 获取可选的文件输入
                file = input('file url (press enter if no file): ').strip()
                
                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                    
                # 构建消息
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式
    
    提供 Web 图形界面，特点：
    - 友好的用户界面
    - 预设查询建议
    - 智能文件管理
    """
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面
        chatbot_config = {
            'prompt.suggestions': [
                '帮我统计一下桌面上有多少个 TXT 文件',
                '列出桌面上所有的 TXT 文件',
                '帮我读取桌面上名为 test.txt 的文件内容',
                '桌面上有哪些文本文件？',
                '读取桌面上最新的 TXT 文件',
                '帮我查看桌面文件情况',
                '桌面上的 TXT 文件都叫什么名字？',
                '读取桌面上的 readme.txt 文件',
                '统计桌面文本文件数量',
                '显示桌面上所有文本文件的列表'
            ]
        }
        
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    # test()           # 测试模式
    # app_tui()        # 终端交互模式
    app_gui()          # 图形界面模式（默认）