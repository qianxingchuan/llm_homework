import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=== offerType 字段分析 ===")

# 加载数据
df_train = pd.read_csv('used_car_train_20200313.csv', sep=' ')

print("1. offerType 字段基本统计:")
print(df_train['offerType'].value_counts())
print(f"\n缺失值数量: {df_train['offerType'].isnull().sum()}")

print("\n2. 各offerType对应的价格统计:")
price_by_offer = df_train.groupby('offerType')['price'].agg(['count', 'mean', 'std', 'min', 'max'])
print(price_by_offer)

print("\n3. offerType与价格的关系分析:")
# 计算每种类型的价格分布
offer_0_prices = df_train[df_train['offerType'] == 0]['price']
offer_1_prices = df_train[df_train['offerType'] == 1]['price']

print(f"offerType=0 (提供) 的价格统计:")
print(f"  数量: {len(offer_0_prices)}")
print(f"  平均价格: {offer_0_prices.mean():.2f}")
print(f"  中位数价格: {offer_0_prices.median():.2f}")
print(f"  标准差: {offer_0_prices.std():.2f}")

print(f"\nofferType=1 (请求) 的价格统计:")
print(f"  数量: {len(offer_1_prices)}")
print(f"  平均价格: {offer_1_prices.mean():.2f}")
print(f"  中位数价格: {offer_1_prices.median():.2f}")
print(f"  标准差: {offer_1_prices.std():.2f}")

# 计算价格差异
price_diff = offer_0_prices.mean() - offer_1_prices.mean()
print(f"\n价格差异 (提供-请求): {price_diff:.2f}")

# 计算相关性
correlation = df_train['offerType'].corr(df_train['price'])
print(f"offerType与价格的相关系数: {correlation:.4f}")

print("\n4. 建议的特征工程策略:")
print("- 将offerType作为分类特征直接使用")
print("- 创建offerType与其他重要特征的交互项")
print("- 考虑为不同offerType类型训练不同的模型")
print("- 将offerType作为模型融合时的分组依据")

# 可视化
plt.figure(figsize=(12, 4))

# 价格分布对比
plt.subplot(1, 2, 1)
plt.hist(offer_0_prices, alpha=0.7, label='提供(0)', bins=50)
plt.hist(offer_1_prices, alpha=0.7, label='请求(1)', bins=50)
plt.xlabel('价格')
plt.ylabel('频次')
plt.title('不同offerType的价格分布')
plt.legend()

# 箱线图
plt.subplot(1, 2, 2)
df_train.boxplot(column='price', by='offerType', ax=plt.gca())
plt.title('不同offerType的价格箱线图')
plt.suptitle('')

plt.tight_layout()
plt.savefig('offer_type_analysis.png', dpi=300, bbox_inches='tight')
print("\n可视化图表已保存为: offer_type_analysis.png") 