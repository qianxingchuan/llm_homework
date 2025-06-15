from vanna.openai import OpenAI_Chat  # 添加缺失的导入
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
import mysql.connector
import time
import os
from openai import OpenAI

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        # 为ChromaDB配置添加必需的_type参数
        chroma_config = {
            '_type': 'chromadb_vector',  # 添加必需的_type参数
            'path': './chroma_db'  # ChromaDB数据存储路径
        }
        
        # 如果有额外配置，合并进去
        if config:
            chroma_config.update({k: v for k, v in config.items() if k not in ['model', 'client']})
        
        # OpenAI配置
        openai_config = {}
        if config:
            if 'model' in config:
                openai_config['model'] = config['model']
            if 'client' in config:
                openai_config['client'] = config['client']
                # 保存client到实例属性，这很重要！
                self.client = config['client']
        
        # 初始化两个基类
        ChromaDB_VectorStore.__init__(self, config=chroma_config)
        OpenAI_Chat.__init__(self, config=openai_config)

# 配置qwen-max-2025-01-25模型
# 请替换为你的实际API Key
api_key = os.environ.get('DASHSCOPE_APP_KEY')
client = OpenAI(
    api_key=api_key,
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'  # 通义千问的兼容接口
)

# 初始化Vanna实例，使用qwen-max-2025-01-25模型
vn = MyVanna(config={
    'model': 'qwen-max-2025-01-25',  # 使用qwen-max-2025-01-25模型
    'client': client
})

# 连接到MySQL数据库（请根据你的实际数据库配置修改）
try:
    vn.connect_to_mysql(
        host='192.168.1.4',  # 数据库主机
        dbname='project-db-001',  # 数据库名
        user='root',  # 用户名
        password='123456',  # 密码
        port=3306  # 端口
    )
    print("成功连接到MySQL数据库")
except Exception as e:
    print(f"数据库连接失败: {e}")
    print("请检查数据库配置信息")

# 训练数据库schema（可选）
def train_database_schema():
    try:
        connection = mysql.connector.connect(
            host='192.168.1.4',
            database='project-db-001',
            user='root',
            password='123456',
            port=3306
        )
        
        cursor = connection.cursor()
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = 'project-db-001'
        """)
        tables = cursor.fetchall()
        
        # 训练每个表的schema
        for (table_name,) in tables:
            try:
                cursor.execute(f"SHOW CREATE TABLE {table_name}")
                _, create_table = cursor.fetchone()
                
                print(f"正在训练表 {table_name} 的schema...")
                vn.train(ddl=create_table)
                
            except Exception as e:
                print(f"训练表 {table_name} 失败: {str(e)}")
                continue
        
        print("Schema训练完成")
        
    except mysql.connector.Error as err:
        print(f"数据库连接错误: {err}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

# 示例查询
def example_queries():
    try:
        # 示例1：自然语言查询
        question = "找出英雄攻击力最高的前5个英雄"
        print(f"\n问题: {question}")
        result = vn.ask(question)
        print(result)

        # 示例2：生成SQL
        sql = vn.generate_sql("找出英雄攻击力最高的前5个英雄")
        print(f"\n生成的SQL: {sql}")
        
        # 示例3：执行SQL
        if sql:
            df = vn.run_sql(sql)
            print(f"查询结果: {df}")
            
    except Exception as e:
        print(f"查询过程中出现错误: {e}")

if __name__ == "__main__":
    print("Vanna with Qwen-Max-2025-01-25 初始化完成")
    
    # 如果需要训练schema，取消下面的注释
    train_database_schema()
    
    # 运行示例查询
    example_queries()