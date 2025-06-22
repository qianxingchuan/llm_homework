import inventory_api 
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_community.llms import Tongyi
from langchain.memory import ConversationBufferMemory
import re
import json
from typing import List, Union, Dict, Any
import os
import dashscope

# 设置通义千问API密钥
api_key = os.environ.get('DASHSCOPE_APP_KEY')
DASHSCOPE_API_KEY = api_key
dashscope.api_key=api_key

inventoryAPI = inventory_api.InventoryAPI()

# 包装函数，确保正确的参数传递和错误处理
def get_overview_inventory_wrapper(sku_code: str) -> str:
    """查询总览库存的包装函数"""
    try:
        result = inventoryAPI.get_overview_inventory_by_sku(sku_code)
        if result['success']:
            data = result['data']
            return json.dumps({
                "totalQty": data['totalQty'],
                "totalOccupied": data['totalOccupied'], 
                "availableQty": data['availableQty']
            }, ensure_ascii=False)
        else:
            return f"查询失败: {result['error']}"
    except Exception as e:
        return f"查询总览库存时出错: {str(e)}"

def get_physical_inventory_wrapper(input_str: str) -> str:
    """查询物理库存的包装函数"""
    try:
        # 解析输入参数
        if ',' in input_str:
            parts = input_str.split(',')
            sku_code = parts[0].strip()
            bin_code = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        else:
            sku_code = input_str.strip()
            bin_code = None
        
        result = inventoryAPI.get_physical_inventory_by_sku_bin(sku_code, bin_code)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"查询物理库存时出错: {str(e)}"

tools=[
    Tool(
        name="逻辑库存查询",
        func=get_overview_inventory_wrapper,
        description="""通过SKU编码查询商品的总览库存信息，包括总库存量、占用量和可用量。
        输入参数：SKU编码（如：榴莲、苹果、香蕉等商品名称或具体的SKU编码）
        返回：JSON格式的库存信息，包含totalQty（总库存）、totalOccupied（占用量）、availableQty（可用量）
        使用示例：输入"榴莲"或"DURIAN001"查询榴莲的总览库存"""
    ),
    Tool(
        name="物理库存查询",
        func=get_physical_inventory_wrapper,
        description="""通过SKU编码查询商品在具体储位的物理库存详细信息。
        输入参数：SKU编码，可选择性加上储位编码（用逗号分隔）
        - 仅SKU：查询该SKU在所有储位的库存
        - SKU,储位编码：查询该SKU在指定储位的库存
        返回：JSON格式的详细库存信息，包含储位、批次、数量等信息
        使用示例：输入"榴莲"查询榴莲在所有储位的库存，或"榴莲,A-01-01"查询榴莲在A-01-01储位的库存"""
    )
]

# 创建工具链
def create_tool_chain():
    """创建工具链"""
    
    # 初始化语言模型
    llm = Tongyi(
        model_name="qwen-max-2025-01-25", 
        dashscope_api_key=DASHSCOPE_API_KEY,
        temperature=0.0  # 设置为0，确保输出格式一致
    )
    
    # 修改提示模板，使格式更加严格
    prompt = PromptTemplate.from_template(
        """你是一个专业的库存管理助手。请严格按照以下格式回答问题：

可用工具：
{tools}

工具名称：{tool_names}

请严格按照以下格式回答，每一步都必须包含：

Question: {input}
Thought: 我需要分析这个问题
Action: [工具名称]
Action Input: [输入参数]
Observation: [工具返回结果]
Thought: 我现在知道最终答案了
Final Answer: [最终答案]

重要规则：
1. Action必须是以下之一：{tool_names}
2. 每次只能使用一个Action
3. 如果需要查询多个商品，请分别调用
4. 必须严格遵循格式，不能省略任何步骤
5. Final Answer前必须有"Thought: 我现在知道最终答案了"

开始：
{agent_scratchpad}"""
    )
    
    # 创建Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # 创建Agent执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,  # 减少迭代次数
        early_stopping_method="generate",
        return_intermediate_steps=True  # 返回中间步骤便于调试
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

# 测试函数
def test_inventory_api():
    """测试inventory_api是否正常工作"""
    print("=== 测试inventory_api直接调用 ===")
    
    # 测试总览库存查询
    print("\n1. 测试总览库存查询:")
    result1 = inventoryAPI.get_overview_inventory(search_sku="榴莲")
    print(f"榴莲总览库存: {result1}")
    
    result2 = inventoryAPI.get_overview_inventory(search_sku="苹果")
    print(f"苹果总览库存: {result2}")
    
    # 测试物理库存查询
    print("\n2. 测试物理库存查询:")
    result3 = inventoryAPI.get_physical_inventory(search_sku="榴莲")
    print(f"榴莲物理库存: {result3}")
    
    print("\n=== API测试完成 ===")

# 简化版本的工具链，使用更简单的提示
def create_simple_tool_chain():
    """创建简化版工具链"""
    
    llm = Tongyi(
        model_name="qwen-max-2025-01-25", 
        dashscope_api_key=DASHSCOPE_API_KEY,
        temperature=0.0
    )
    
    # 使用更简单的提示模板
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor

# 简化版处理函数
def process_task_simple(task_description):
    """使用简化版工具链处理任务"""
    try:
        agent_executor = create_simple_tool_chain()
        response = agent_executor.invoke({"input": task_description})
        return response["output"]
    except Exception as e:
        return f"处理任务时出错: {str(e)}"
    
# 示例用法
if __name__ == "__main__":
    # 首先测试API是否正常
    test_inventory_api()
    
    print("\n" + "="*50)
    print("开始测试简化版工具链...")
    print("="*50)
    
    # 使用简化版工具链测试
    task1 = "查询榴莲的库存信息"
    print(f"\n简单任务测试: {task1}")
    print("结果:", process_task_simple(task1))
    
    # 复杂任务测试
    task2 = "我现在有一个出库单，出库单希望可以出库榴莲20个，苹果500个，帮我看一下我的库存是否允许接单。判断逻辑是，只要总览库存有足够的可用库存，即满足条件"
    print(f"\n复杂任务测试: {task2}")
    print("结果:", process_task_simple(task2))
