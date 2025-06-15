from vanna.openai import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
import mysql.connector
import time
import os
from openai import OpenAI

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        # 分离配置，只将OpenAI客户端传递给OpenAI_Chat
        chroma_config = config.copy() if config else {}
        openai_config = config.copy() if config else {}
        
        # 保存client到实例属性
        if config and 'client' in config:
            self.client = config['client']
        else:
            self.client = None
        
        # 从chroma_config中移除client
        if 'client' in chroma_config:
            del chroma_config['client']
            
        # 初始化两个基类
        ChromaDB_VectorStore.__init__(self, config=chroma_config)
        OpenAI_Chat.__init__(self, config=openai_config)

# 创建OpenAI客户端
# 从环境变量获取 dashscope 的 API Key
api_key = 'sk-HsggDrwOjUrq6QYZ6fOLFzoPkK3R8eyHx8gf75YlGA8KiSyq'
client = OpenAI(
    api_key=api_key,
    base_url='https://api.fe8.cn/v1'
)

# 初始化Vanna实例
vn = MyVanna(config={
    'model': 'gpt-4o-mini', 
    # 'model': 'qwen-max-2025-01-25', 
    'client': client
})

vn.connect_to_mysql(host='192.168.1.4', 
                    dbname='project-db-001', user='root', password='123456', port=3306)

# 连接到MySQL数据库
try:
    connection = mysql.connector.connect(
        host='192.168.1.4',
        database='project-db-001',
        user='root',
        password='123456',
        port=3306
    )
    print("成功连接到MySQL数据库")
    
    # 获取所有表名
    cursor = connection.cursor()
    cursor.execute("""
        SELECT TABLE_NAME 
        FROM information_schema.TABLES 
        WHERE TABLE_SCHEMA = 'action'
    """)
    tables = cursor.fetchall()
    
    # 训练每个表的schema
    for (table_name,) in tables:
        try:
            # 获取表的创建语句
            cursor.execute(f"SHOW CREATE TABLE {table_name}")
            _, create_table = cursor.fetchone()
            
            print(f"正在训练表 {table_name} 的schema...")
            vn.train(ddl=create_table)
            
        except Exception as e:
            print(f"训练表 {table_name} 失败: {str(e)}")
            continue
    
    print("Schema训练完成")
    
    # 示例：使用Vanna进行自然语言查询
    question = "找出英雄攻击力最高的前5个英雄"
    #print(f"\n问题: {question}")
    vn.ask(question)
    #vn.ask("查询heros表中 英雄攻击力前5名的英雄")
    # sql=vn.generate_sql("查询heros表中 英雄攻击力前5名的英雄")
    # print('sql=', sql)
    # df = vn.run_sql(sql)
    # print('df=', df)
    
except mysql.connector.Error as err:
    print(f"数据库连接错误: {err}")
finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL连接已关闭")


# In[ ]:


# #help(vn.ask)
# #vn.ask("查询heros表中 英雄攻击力前5名的英雄")
# sql=vn.generate_sql("查询heros表中 英雄攻击力前5名的英雄")
# print('sql=', sql)
# vn.run_sql(sql)

