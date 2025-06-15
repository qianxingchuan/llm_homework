#!/usr/bin/env python
# coding: utf-8

# ## 使用DeepSeek进行数据表的查询

# In[1]:


from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor

db_user = "root"
db_password = "123456"
#db_host = "localhost:3306"
db_host = "192.168.1.4:3306"
db_name = "project-db-001"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
db


# In[3]:


from langchain.chat_models import ChatOpenAI
import os

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_APP_KEY')

# 通过LLM => 撰写SQL
llm = ChatOpenAI(
    temperature=0.01,
    # model="deepseek-v3",  
    model = "qwen-max-2025-01-25",
    # openai_api_key = "sk-9846f14a2104490b960adbf5c5b3b32e",
    # openai_api_base="https://api.deepseek.com"
    openai_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key  = api_key
)

# 需要设置llm
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# SQL智能体：给它目标，它自己会进行规划，最终把结果给你
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)


# In[4]:


# Task: 描述数据表
agent_executor.run("描述与订单相关的表及其关系")


# In[7]:


# 这个任务，实际上数据库中 没有HeroDetails表
agent_executor.run("描述HeroDetails表")


# In[3]:


agent_executor.run("描述Hero表")


# In[7]:


agent_executor.run("找出英雄攻击力最高的前5个英雄")

