import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取数据
current_dir = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(current_dir, 'hospital_bed_usage_data.xlsx')
df = pd.read_excel(excel_file)

# 创建图表目录
output_dir = os.path.join(current_dir, 'charts')
os.makedirs(output_dir, exist_ok=True)

# 1. 医院整体使用率箱线图
plt.figure(figsize=(15, 6))
sns.boxplot(x='hospital_name', y='occupancy_rate', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('各医院病床使用率分布')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_hospital_boxplot.png'))
plt.close()

# 2. 科室平均使用率热力图
dept_hospital_pivot = df.pivot_table(
    values='occupancy_rate',
    index='department_name',
    columns='hospital_district',
    aggfunc='mean'
).round(2)

plt.figure(figsize=(10, 12))
sns.heatmap(dept_hospital_pivot, annot=True, fmt='.1f', cmap='YlOrRd', center=80)
plt.title('各区域各科室平均病床使用率热力图')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_department_district_heatmap.png'))
plt.close()

# 3. 各科室使用率箱线图
plt.figure(figsize=(15, 8))
sns.boxplot(x='department_name', y='occupancy_rate', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('各科室病床使用率分布')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_department_boxplot.png'))
plt.close()

# 4. 医院区域使用率条形图
district_stats = df.groupby('hospital_district')['occupancy_rate'].agg(['mean', 'std']).round(2)
plt.figure(figsize=(10, 6))
district_stats['mean'].plot(kind='bar', yerr=district_stats['std'], capsize=5)
plt.title('各区域平均病床使用率')
plt.ylabel('使用率 (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_district_barplot.png'))
plt.close()

# 5. 时间趋势图（取每个时间点的平均值）
time_trend = df.groupby('timestamp')['occupancy_rate'].mean()
plt.figure(figsize=(12, 6))
time_trend.plot(kind='line', marker='o')
plt.title('病床使用率时间趋势')
plt.xlabel('时间')
plt.ylabel('平均使用率 (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '5_time_trend.png'))
plt.close()

# 6. 综合分析图表
fig = plt.figure(figsize=(20, 15))
fig.suptitle('医院病床使用情况综合分析', fontsize=16, y=0.95)

# 6.1 区域使用率箱线图 (左上)
ax1 = plt.subplot(2, 2, 1)
sns.boxplot(x='hospital_district', y='occupancy_rate', data=df, ax=ax1)
ax1.set_title('各区域病床使用率分布')
ax1.set_xlabel('医院区域')
ax1.set_ylabel('使用率 (%)')

# 6.2 科室平均使用率热力图 (右上)
ax2 = plt.subplot(2, 2, 2)
sns.heatmap(dept_hospital_pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
            center=80, ax=ax2, cbar_kws={'label': '使用率 (%)'})
ax2.set_title('各区域各科室平均病床使用率')
ax2.set_xlabel('医院区域')
ax2.set_ylabel('科室名称')

# 6.3 各科室平均使用率条形图 (左下)
dept_stats = df.groupby('department_name')['occupancy_rate'].agg(['mean', 'std']).round(2)
dept_stats = dept_stats.sort_values('mean', ascending=True)

ax3 = plt.subplot(2, 2, 3)
dept_stats['mean'].plot(kind='barh', yerr=dept_stats['std'], 
                       capsize=5, ax=ax3)
ax3.set_title('各科室平均病床使用率')
ax3.set_xlabel('使用率 (%)')
ax3.grid(True, axis='x')

# 6.4 时间趋势图 (右下)
time_trend = df.groupby(['timestamp', 'hospital_district'])['occupancy_rate'].mean().unstack()
ax4 = plt.subplot(2, 2, 4)
time_trend.plot(marker='o', ax=ax4)
ax4.set_title('各区域病床使用率时间趋势')
ax4.set_xlabel('时间')
ax4.set_ylabel('平均使用率 (%)')
ax4.grid(True)
ax4.legend(title='医院区域')

# 调整布局
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '6_comprehensive_analysis.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

print("图表已生成在 'charts' 目录下：")
print("1. 各医院病床使用率分布 (1_hospital_boxplot.png)")
print("2. 各区域各科室平均病床使用率热力图 (2_department_district_heatmap.png)")
print("3. 各科室病床使用率分布 (3_department_boxplot.png)")
print("4. 各区域平均病床使用率 (4_district_barplot.png)")
print("5. 病床使用率时间趋势 (5_time_trend.png)")
print("6. 综合分析图表 (6_comprehensive_analysis.png)") 