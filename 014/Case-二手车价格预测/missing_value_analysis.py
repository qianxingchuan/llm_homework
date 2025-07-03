import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== 二手车数据集缺失值分析 ===")

# 1. 加载数据
print("\n1. 加载数据...")
df_train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
df_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集形状: {df_train.shape}")
print(f"测试集形状: {df_test.shape}")

# 2. 分析训练集缺失值
print("\n2. 训练集缺失值分析...")

# 计算缺失值
missing_train = df_train.isnull().sum()
missing_train_pct = (missing_train / len(df_train)) * 100

# 创建缺失值统计DataFrame
missing_stats_train = pd.DataFrame({
    '字段名': missing_train.index,
    '缺失数量': missing_train.values,
    '缺失占比(%)': missing_train_pct.values
}).sort_values('缺失占比(%)', ascending=False)

print("训练集缺失值统计:")
print(missing_stats_train)

# 3. 分析测试集缺失值
print("\n3. 测试集缺失值分析...")

missing_test = df_test.isnull().sum()
missing_test_pct = (missing_test / len(df_test)) * 100

missing_stats_test = pd.DataFrame({
    '字段名': missing_test.index,
    '缺失数量': missing_test.values,
    '缺失占比(%)': missing_test_pct.values
}).sort_values('缺失占比(%)', ascending=False)

print("测试集缺失值统计:")
print(missing_stats_test)

# 4. 对比分析
print("\n4. 训练集与测试集缺失值对比...")

# 合并两个数据集的缺失值信息
comparison_df = pd.DataFrame({
    '字段名': missing_stats_train['字段名'],
    '训练集缺失占比(%)': missing_stats_train['缺失占比(%)'],
    '测试集缺失占比(%)': missing_stats_test['缺失占比(%)']
})

# 计算差异
comparison_df['差异(%)'] = comparison_df['测试集缺失占比(%)'] - comparison_df['训练集缺失占比(%)']

print("缺失值对比分析:")
print(comparison_df)

# 5. 可视化缺失值分布
print("\n5. 生成缺失值可视化图表...")

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# 训练集缺失值条形图
missing_train_filtered = missing_stats_train[missing_stats_train['缺失占比(%)'] > 0]
axes[0, 0].barh(missing_train_filtered['字段名'], missing_train_filtered['缺失占比(%)'])
axes[0, 0].set_title('训练集缺失值占比')
axes[0, 0].set_xlabel('缺失占比 (%)')
axes[0, 0].invert_yaxis()

# 测试集缺失值条形图
missing_test_filtered = missing_stats_test[missing_stats_test['缺失占比(%)'] > 0]
axes[0, 1].barh(missing_test_filtered['字段名'], missing_test_filtered['缺失占比(%)'])
axes[0, 1].set_title('测试集缺失值占比')
axes[0, 1].set_xlabel('缺失占比 (%)')
axes[0, 1].invert_yaxis()

# 缺失值热力图
missing_matrix_train = df_train.isnull()
missing_matrix_test = df_test.isnull()

# 计算每列的缺失比例
missing_heatmap_data = pd.DataFrame({
    '训练集': missing_matrix_train.sum() / len(df_train) * 100,
    '测试集': missing_matrix_test.sum() / len(df_test) * 100
})

# 只显示有缺失值的字段
missing_heatmap_data = missing_heatmap_data[(missing_heatmap_data['训练集'] > 0) | (missing_heatmap_data['测试集'] > 0)]

sns.heatmap(missing_heatmap_data.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 0])
axes[1, 0].set_title('缺失值热力图')

# 缺失值对比散点图
axes[1, 1].scatter(comparison_df['训练集缺失占比(%)'], comparison_df['测试集缺失占比(%)'], alpha=0.7)
axes[1, 1].plot([0, comparison_df['训练集缺失占比(%)'].max()], [0, comparison_df['测试集缺失占比(%)'].max()], 'r--', alpha=0.5)
axes[1, 1].set_xlabel('训练集缺失占比 (%)')
axes[1, 1].set_ylabel('测试集缺失占比 (%)')
axes[1, 1].set_title('训练集 vs 测试集缺失值对比')

# 添加字段标签
for idx, row in comparison_df.iterrows():
    if row['训练集缺失占比(%)'] > 0 or row['测试集缺失占比(%)'] > 0:
        axes[1, 1].annotate(row['字段名'], 
                           (row['训练集缺失占比(%)'], row['测试集缺失占比(%)']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.savefig('missing_value_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 详细分析特定字段
print("\n6. 特定字段详细分析...")

# 分析有缺失值的字段
fields_with_missing = missing_stats_train[missing_stats_train['缺失占比(%)'] > 0]['字段名'].tolist()

print("有缺失值的字段详细分析:")
for field in fields_with_missing:
    print(f"\n字段: {field}")
    
    # 训练集分析
    train_missing = missing_stats_train[missing_stats_train['字段名'] == field].iloc[0]
    print(f"  训练集: {train_missing['缺失数量']} 条缺失 ({train_missing['缺失占比(%)']:.2f}%)")
    
    # 测试集分析
    test_missing = missing_stats_test[missing_stats_test['字段名'] == field].iloc[0]
    print(f"  测试集: {test_missing['缺失数量']} 条缺失 ({test_missing['缺失占比(%)']:.2f}%)")
    
    # 数据类型
    print(f"  数据类型: {df_train[field].dtype}")
    
    # 非缺失值的统计信息
    if df_train[field].dtype in ['int64', 'float64']:
        non_missing_values = df_train[field].dropna()
        print(f"  非缺失值统计: 均值={non_missing_values.mean():.2f}, 中位数={non_missing_values.median():.2f}")
    else:
        value_counts = df_train[field].value_counts()
        print(f"  非缺失值分布: {value_counts.head(3).to_dict()}")

# 7. 缺失值模式分析
print("\n7. 缺失值模式分析...")

# 检查是否存在同时缺失多个字段的情况
missing_pattern = df_train[fields_with_missing].isnull()
missing_pattern_sum = missing_pattern.sum(axis=1)

print("同时缺失多个字段的统计:")
pattern_counts = missing_pattern_sum.value_counts().sort_index()
for pattern, count in pattern_counts.items():
    percentage = (count / len(df_train)) * 100
    print(f"  同时缺失 {pattern} 个字段: {count} 条记录 ({percentage:.2f}%)")

# 8. 生成详细报告
print("\n8. 生成详细报告...")

with open('missing_value_report.txt', 'w', encoding='utf-8') as f:
    f.write("二手车数据集缺失值分析报告\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. 数据集概览\n")
    f.write(f"训练集样本数: {len(df_train)}\n")
    f.write(f"测试集样本数: {len(df_test)}\n")
    f.write(f"总字段数: {len(df_train.columns)}\n\n")
    
    f.write("2. 训练集缺失值统计\n")
    f.write(missing_stats_train.to_string(index=False))
    f.write("\n\n")
    
    f.write("3. 测试集缺失值统计\n")
    f.write(missing_stats_test.to_string(index=False))
    f.write("\n\n")
    
    f.write("4. 缺失值对比分析\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("5. 缺失值模式分析\n")
    for pattern, count in pattern_counts.items():
        percentage = (count / len(df_train)) * 100
        f.write(f"同时缺失 {pattern} 个字段: {count} 条记录 ({percentage:.2f}%)\n")

print("详细报告已保存到: missing_value_report.txt")
print("可视化图表已保存到: missing_value_analysis.png")

# 9. 总结
print("\n=== 缺失值分析总结 ===")
print(f"总字段数: {len(df_train.columns)}")
print(f"有缺失值的字段数: {len(fields_with_missing)}")
print(f"缺失值最多的字段: {missing_stats_train.iloc[0]['字段名']} ({missing_stats_train.iloc[0]['缺失占比(%)']:.2f}%)")
print(f"缺失值最少的字段: {missing_stats_train[missing_stats_train['缺失占比(%)'] > 0].iloc[-1]['字段名']} ({missing_stats_train[missing_stats_train['缺失占比(%)'] > 0].iloc[-1]['缺失占比(%)']:.2f}%)")

print("\n缺失值分析完成！") 