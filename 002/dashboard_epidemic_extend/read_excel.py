import pandas as pd

# 读取Excel文件
file_path = "香港各区疫情数据_20250322.xlsx"
df = pd.read_excel(file_path)

# 显示列名（字段）
print("文件字段列表：")
for col in df.columns:
    print(f"- {col}")

print("\n前20行数据：")
print(df.head(20))