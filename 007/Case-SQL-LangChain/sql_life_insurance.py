#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor

db_user = "student123"
db_password = "student321"
db_host = "rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306"
db_name = "life_insurance"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
#engine = create_engine('mysql+mysqlconnector://root:passw0rdcc4@localhost:3306/wucai')
db


# In[2]:


from langchain.chat_models import ChatOpenAI
import os

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_APP_KEY')

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

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Task1
result = agent_executor.invoke("获取所有客户的姓名和联系电话")
print(result)


# In[3]:


result=agent_executor.run("查询所有未支付保费的保单号和客户姓名。")
print(result)

# In[4]:


agent_executor.run("找出所有理赔金额大于10000元的理赔记录，并列出相关客户的姓名和联系电话。")

