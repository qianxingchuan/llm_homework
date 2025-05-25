import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
import matplotlib.font_manager as fm
import matplotlib as mpl
import platform

# 根据操作系统设置中文字体
system = platform.system()
if system == 'Windows':
    # Windows系统使用微软雅黑或SimHei
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # 使用黑体
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        # 尝试使用其他常见中文字体
        for font in ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']:
            try:
                mpl.rc('font', family=font)
                break
            except:
                continue
elif system == 'Darwin':  # macOS
    # macOS系统使用苹方或华文黑体
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'STHeiti', 'Heiti TC', 'PingFang SC']
else:  # Linux
    # Linux系统使用文泉驿等字体
    plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

# 确保负号正确显示
plt.rcParams['axes.unicode_minus'] = False

# 设置全局图表样式
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 创建输出目录
output_dir = "eda_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取Excel文件
print("正在读取数据...")
file_path = "香港各区疫情数据_20250322.xlsx"
df = pd.read_excel(file_path)

# 确保日期列是日期类型
df['报告日期'] = pd.to_datetime(df['报告日期'])

# 1. 基本数据信息
print("\n1. 基本数据信息")
print("-" * 80)
print(f"数据集形状: {df.shape}")
print(f"数据时间范围: {df['报告日期'].min()} 至 {df['报告日期'].max()}")
print(f"总记录数: {len(df)}")
print(f"地区数量: {df['地区名称'].nunique()}")

# 保存基本信息到文件
with open(f"{output_dir}/1_基本信息.txt", "w", encoding="utf-8") as f:
    f.write(f"数据集形状: {df.shape}\n")
    f.write(f"数据时间范围: {df['报告日期'].min()} 至 {df['报告日期'].max()}\n")
    f.write(f"总记录数: {len(df)}\n")
    f.write(f"地区数量: {df['地区名称'].nunique()}\n")
    f.write("\n地区列表:\n")
    for district in sorted(df['地区名称'].unique()):
        f.write(f"- {district}\n")

# 2. 数据摘要统计
print("\n2. 数据摘要统计")
print("-" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
summary = df[numeric_cols].describe().T
summary['缺失值'] = df[numeric_cols].isna().sum()
summary['缺失率'] = df[numeric_cols].isna().mean().round(4) * 100
print(summary)

# 保存摘要统计到CSV
summary.to_csv(f"{output_dir}/2_数据摘要统计.csv", encoding="utf-8-sig")

# 3. 时间序列分析 - 全香港每日新增确诊趋势
print("\n3. 时间序列分析")
print("-" * 80)

# 按日期汇总全香港数据
daily_total = df.groupby('报告日期').agg({
    '新增确诊': 'sum',
    '新增康复': 'sum',
    '新增死亡': 'sum',
    '累计确诊': 'max',
    '累计康复': 'max',
    '累计死亡': 'max'
}).reset_index()

# 绘制每日新增确诊趋势
plt.figure(figsize=(15, 6))
plt.plot(daily_total['报告日期'], daily_total['新增确诊'], marker='o', linestyle='-', color='#FF9999', alpha=0.7, markersize=2)
plt.title('香港每日新增确诊病例趋势', fontsize=16, pad=20)
plt.xlabel('日期', fontsize=12)
plt.ylabel('新增确诊病例数', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/3_1_每日新增确诊趋势.png", dpi=300, bbox_inches='tight')
plt.close()

# 绘制累计确诊、康复和死亡趋势
plt.figure(figsize=(15, 6))
plt.plot(daily_total['报告日期'], daily_total['累计确诊'], label='累计确诊', color='#FF6666', linewidth=2)
plt.plot(daily_total['报告日期'], daily_total['累计康复'], label='累计康复', color='#66CC66', linewidth=2)
plt.plot(daily_total['报告日期'], daily_total['累计死亡'], label='累计死亡', color='#666666', linewidth=2)
plt.title('香港累计确诊、康复和死亡病例趋势', fontsize=16, pad=20)
plt.xlabel('日期', fontsize=12)
plt.ylabel('病例数', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/3_2_累计病例趋势.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. 地区分析
print("\n4. 地区分析")
print("-" * 80)

# 计算每个地区的总确诊病例
district_total = df.groupby('地区名称').agg({
    '新增确诊': 'sum',
    '新增康复': 'sum',
    '新增死亡': 'sum',
    '人口': 'first'  # 每个地区的人口是固定的
}).reset_index()

# 计算每10万人确诊率
district_total['每10万人确诊率'] = (district_total['新增确诊'] / district_total['人口']) * 100000
district_total = district_total.sort_values('每10万人确诊率', ascending=False)

# 绘制地区确诊病例对比
plt.figure(figsize=(14, 8))
sns.barplot(x='地区名称', y='新增确诊', data=district_total.sort_values('新增确诊', ascending=False))
plt.title('香港各地区总确诊病例对比', fontsize=16, pad=20)
plt.xlabel('地区', fontsize=12)
plt.ylabel('总确诊病例数', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/4_1_地区确诊病例对比.png", dpi=300, bbox_inches='tight')
plt.close()

# 绘制地区每10万人确诊率对比
plt.figure(figsize=(14, 8))
sns.barplot(x='地区名称', y='每10万人确诊率', data=district_total)
plt.title('香港各地区每10万人确诊率对比', fontsize=16, pad=20)
plt.xlabel('地区', fontsize=12)
plt.ylabel('每10万人确诊率', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/4_2_地区每10万人确诊率对比.png", dpi=300, bbox_inches='tight')
plt.close()

# 保存地区分析结果到CSV
district_total.to_csv(f"{output_dir}/4_地区分析结果.csv", index=False, encoding="utf-8-sig")

# 5. 风险等级分析
print("\n5. 风险等级分析")
print("-" * 80)

# 统计不同风险等级的记录数
risk_counts = df['风险等级'].value_counts().reset_index()
risk_counts.columns = ['风险等级', '记录数']
print(risk_counts)

# 绘制风险等级分布饼图
plt.figure(figsize=(10, 8))
colors = ['#FF9999', '#66B2FF', '#99FF99']
plt.pie(risk_counts['记录数'], labels=risk_counts['风险等级'], autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('香港各区风险等级分布', fontsize=16, pad=20)
plt.axis('equal')
plt.tight_layout()
plt.savefig(f"{output_dir}/5_风险等级分布.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. 相关性分析
print("\n6. 相关性分析")
print("-" * 80)

# 计算数值变量之间的相关性
corr_matrix = df[['新增确诊', '累计确诊', '现存确诊', '新增康复', '累计康复', 
                  '新增死亡', '累计死亡', '发病率(每10万人)', '人口']].corr()
print(corr_matrix.round(2))

# 绘制相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('变量相关性热力图', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(f"{output_dir}/6_相关性热力图.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. 时间序列热力图 - 展示不同地区随时间的确诊病例变化
print("\n7. 时间序列热力图")
print("-" * 80)

# 选择数据的一个子集进行可视化（例如，每月的第一天）
df['月份'] = df['报告日期'].dt.strftime('%Y-%m')
monthly_data = df.groupby(['月份', '地区名称'])['新增确诊'].sum().reset_index()
monthly_pivot = monthly_data.pivot(index='地区名称', columns='月份', values='新增确诊')

plt.figure(figsize=(16, 10))
sns.heatmap(monthly_pivot, cmap='YlOrRd', linewidths=0.5, annot=False)
plt.title('香港各地区月度新增确诊热力图', fontsize=16, pad=20)
plt.xlabel('月份', fontsize=12)
plt.ylabel('地区', fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/7_地区月度确诊热力图.png", dpi=300, bbox_inches='tight')
plt.close()

# 8. 箱线图 - 各地区新增确诊分布
print("\n8. 箱线图分析")
print("-" * 80)

plt.figure(figsize=(16, 10))
sns.boxplot(x='地区名称', y='新增确诊', data=df)
plt.title('香港各地区新增确诊分布箱线图', fontsize=16, pad=20)
plt.xlabel('地区', fontsize=12)
plt.ylabel('新增确诊病例数', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/8_地区新增确诊箱线图.png", dpi=300, bbox_inches='tight')
plt.close()

# 9. 高风险时期分析
print("\n9. 高风险时期分析")
print("-" * 80)

# 找出全香港新增确诊病例最多的前10天
top_days = daily_total.sort_values('新增确诊', ascending=False).head(10)
print("全香港新增确诊病例最多的10天：")
print(top_days[['报告日期', '新增确诊']])

# 绘制高风险时期柱状图
plt.figure(figsize=(14, 8))
sns.barplot(x=top_days['报告日期'].dt.strftime('%Y-%m-%d'), y=top_days['新增确诊'])
plt.title('香港新增确诊病例最多的10天', fontsize=16, pad=20)
plt.xlabel('日期', fontsize=12)
plt.ylabel('新增确诊病例数', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/9_高风险时期分析.png", dpi=300, bbox_inches='tight')
plt.close()

# 10. 地区风险变化分析
print("\n10. 地区风险变化分析")
print("-" * 80)

# 计算每个地区每个风险等级的天数
district_risk = df.groupby(['地区名称', '风险等级']).size().reset_index(name='天数')
district_risk_pivot = district_risk.pivot(index='地区名称', columns='风险等级', values='天数').fillna(0)

# 如果有多个风险等级，绘制堆叠柱状图
if len(df['风险等级'].unique()) > 1:
    plt.figure(figsize=(14, 8))
    district_risk_pivot.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('香港各地区不同风险等级天数', fontsize=16, pad=20)
    plt.xlabel('地区', fontsize=12)
    plt.ylabel('天数', fontsize=12)
    plt.legend(title='风险等级', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/10_地区风险变化分析.png", dpi=300, bbox_inches='tight')
    plt.close()

# 保存地区风险变化分析结果到CSV
district_risk_pivot.to_csv(f"{output_dir}/10_地区风险变化分析.csv", encoding="utf-8-sig")

print(f"\nEDA分析完成！结果已保存到 {output_dir} 目录")