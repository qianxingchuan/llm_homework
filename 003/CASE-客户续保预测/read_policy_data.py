import pandas as pd
import os

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建Excel文件的完整路径
excel_path = os.path.join(current_dir, 'policy_data.xlsx')
# 读取Excel文件
df = pd.read_excel(excel_path)

# 显示所有列名
print("数据字段列表：")
print(df.columns.tolist())
print("\n前5行完整数据：")
# 设置显示所有列
pd.set_option('display.max_columns', None)
# 设置显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('display.max_colwidth', 100)
# 显示前5行数据
print(df.head())