import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# 读取Excel文件
# 文件路径
currentPath = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(currentPath, '香港各区疫情数据_20250322.xlsx')
df = pd.read_excel(file_path)

# 展示前20行数据
print("前20行数据:")
print(df.head(20))

# 展示数据结构
print("\n数据结构:")
print(df.info())

# 计算总新增确诊和累计确诊
total_new_confirmed = df['新增确诊'].sum()
total_cumulative_confirmed = df['累计确诊'].max()

# 按日期分组计算每日新增确诊和累计确诊
daily_confirmed = df.groupby('报告日期')[['新增确诊', '累计确诊']].sum().reset_index()

# 显示每日新增确诊和累计确诊
print("\n每日新增与累计确诊数据:")
print(daily_confirmed.head())

# 格式化报告日期为yyyy-MM-dd格式
daily_confirmed['报告日期'] = pd.to_datetime(daily_confirmed['报告日期']).dt.strftime('%Y-%m-%d')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 绘制每日新增确诊和累计确诊的折线图
plt.figure(figsize=(14, 7))
plt.plot(daily_confirmed['报告日期'], daily_confirmed['新增确诊'], label='每日新增确诊', marker='o')
plt.plot(daily_confirmed['报告日期'], daily_confirmed['累计确诊'], label='累计确诊', marker='o')

plt.xlabel('报告日期')
plt.ylabel('人数')
plt.title('每日新增与累计确诊数据')
plt.legend()
plt.grid(True)

# 调整x轴刻度密度
plt.xticks(daily_confirmed['报告日期'][::7], rotation=45)  # 每7天显示一个刻度

plt.tight_layout()

# 保存图表到与Excel文件相同的目录下
output_path = os.path.join(currentPath, 'daily_confirmed_trend.png')
plt.savefig(output_path)
print(f"\n图表已保存至: {os.path.abspath(output_path)}")

plt.show()