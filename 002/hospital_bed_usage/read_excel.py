import pandas as pd
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(current_dir, 'hospital_bed_usage_data.xlsx')

# 读取Excel文件
df = pd.read_excel(excel_file)

# 1. 计算各医院的平均病床使用率
hospital_occupancy = df.groupby(['hospital_id', 'hospital_name'])[['occupancy_rate']].agg({
    'occupancy_rate': ['mean', 'count']
}).round(2)

hospital_occupancy.columns = ['平均占用率', '样本数']
hospital_occupancy = hospital_occupancy.reset_index()

# 2. 计算各医院各科室的平均病床使用率
dept_occupancy = df.groupby(['hospital_id', 'hospital_name', 'department_id', 'department_name'])[['occupancy_rate']].agg({
    'occupancy_rate': ['mean', 'count']
}).round(2)

dept_occupancy.columns = ['平均占用率', '样本数']
dept_occupancy = dept_occupancy.reset_index()

# 打印结果
print("\n=== 各医院病床平均使用率 ===")
print(hospital_occupancy.to_string(index=False))

print("\n\n=== 各医院各科室病床平均使用率 ===")
print(dept_occupancy.to_string(index=False))

# 显示字段信息
print("\n=== 数据字段信息 ===")
print("\n字段列表:")
for column in df.columns:
    print(f"- {column}")

print("\n=== 数据类型信息 ===")
print(df.dtypes)

print("\n=== 前20行数据 ===")
print(df.head(20)) 