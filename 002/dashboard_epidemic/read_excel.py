import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取Excel文件
excel_file = '香港各区疫情数据_20250322.xlsx'
df = pd.read_excel(excel_file)

# 按报告日期分组计算全香港每日数据
daily_stats = df.groupby('报告日期').agg({
    '新增确诊': 'sum',
    '累计确诊': 'max',
    '新增康复': 'sum',
    '新增死亡': 'sum'
}).reset_index()

# 创建图表
plt.figure(figsize=(15, 10))

# 设置日期刻度间隔
def set_date_ticks(ax, dates, interval=15):
    # 选择要显示的日期索引
    tick_indices = np.arange(0, len(dates), interval)
    # 设置刻度位置和标签
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([dates.iloc[i] for i in tick_indices], rotation=45)

# 1. 最近30天新增确诊病例趋势
ax1 = plt.subplot(2, 2, 1)
last_30_days = daily_stats.tail(30)
plt.plot(range(len(last_30_days)), last_30_days['新增确诊'], color='red', linewidth=2, marker='o')
plt.title('最近30天新增确诊病例趋势')
set_date_ticks(ax1, last_30_days['报告日期'], interval=5)  # 每5天显示一个刻度
plt.grid(True, alpha=0.3)

# 2. 累计确诊病例趋势
ax2 = plt.subplot(2, 2, 2)
plt.plot(range(len(daily_stats)), daily_stats['累计确诊'], color='blue', linewidth=2)
plt.title('累计确诊病例趋势')
set_date_ticks(ax2, daily_stats['报告日期'])
plt.grid(True, alpha=0.3)

# 3. 最近30天新增确诊病例
ax3 = plt.subplot(2, 2, 3)
plt.bar(range(len(last_30_days)), last_30_days['新增确诊'], color='orange')
set_date_ticks(ax3, last_30_days['报告日期'], interval=5)  # 最近30天数据，每5天显示一个刻度
plt.title('最近30天新增确诊病例')
plt.grid(True, alpha=0.3)

# 4. 各区累计确诊情况（饼图）
ax4 = plt.subplot(2, 2, 4)
latest_date = df['报告日期'].max()
district_cases = df[df['报告日期'] == latest_date].sort_values('累计确诊', ascending=False)
# 过滤掉确诊数为0的地区
district_cases = district_cases[district_cases['累计确诊'] > 0]

# 计算总确诊数和比例
total_cases = district_cases['累计确诊'].sum()
district_cases['比例'] = district_cases['累计确诊'] / total_cases * 100

# 将小于5%的区域归类为"其他地区"
threshold = 2
main_districts = district_cases[district_cases['比例'] >= threshold]
other_districts = district_cases[district_cases['比例'] < threshold]

# 准备饼图数据
if len(other_districts) > 0:
    pie_data = pd.concat([
        main_districts,
        pd.DataFrame([{
            '地区名称': '其他地区',
            '累计确诊': other_districts['累计确诊'].sum(),
            '比例': other_districts['比例'].sum()
        }])
    ])
else:
    pie_data = main_districts

# 准备颜色
colors = plt.cm.Pastel1(np.linspace(0, 1, len(pie_data)))

# 绘制饼图
wedges, texts, autotexts = plt.pie(pie_data['累计确诊'],
                                  labels=pie_data['地区名称'],
                                  colors=colors,
                                  autopct='%1.1f%%',
                                  startangle=90,
                                  counterclock=False)

# 添加图例，显示确诊数据
legend_labels = [f"{row['地区名称']}: {row['累计确诊']:,.0f}例" 
                for _, row in pie_data.iterrows()]
plt.legend(wedges, legend_labels,
          title="确诊病例数",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('各区累计确诊情况分布')

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('疫情数据分析图表.png', dpi=300, bbox_inches='tight')

# 显示一些统计数据
print("\n=== 全香港疫情统计 ===")
print(f"总新增确诊病例：{daily_stats['新增确诊'].sum():,.0f}")
print(f"最新累计确诊数：{daily_stats['累计确诊'].iloc[-1]:,.0f}")
print(f"平均每日新增：{daily_stats['新增确诊'].mean():.1f}")

# 显示最近30天统计
print("\n=== 最近30天统计 ===")
print(f"30天内新增确诊病例：{last_30_days['新增确诊'].sum():,.0f}")
print(f"30天内平均每日新增：{last_30_days['新增确诊'].mean():.1f}")

# 显示各区详细数据
print("\n=== 各区确诊情况详细数据 ===")
print(district_cases[['地区名称', '累计确诊', '比例']].to_string(float_format=lambda x: '{:.1f}'.format(x) if isinstance(x, float) else str(x))) 