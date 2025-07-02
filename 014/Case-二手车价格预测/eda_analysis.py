import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("正在读取数据...")
df_train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
df_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集形状: {df_train.shape}")
print(f"测试集形状: {df_test.shape}")

# 1. 基本信息分析
print("\n=== 1. 基本信息分析 ===")
print("训练集列名:", list(df_train.columns))
print("测试集列名:", list(df_test.columns))

# 2. 数据类型和缺失值分析
print("\n=== 2. 数据类型和缺失值分析 ===")
print("训练集数据类型:")
print(df_train.dtypes)
print("\n训练集缺失值统计:")
missing_train = df_train.isnull().sum()
print(missing_train[missing_train > 0])

# 3. 目标变量分析
print("\n=== 3. 目标变量(price)分析 ===")
print("价格统计信息:")
print(df_train['price'].describe())

plt.figure(figsize=(15, 5))

# 价格分布直方图
plt.subplot(1, 3, 1)
plt.hist(df_train['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('价格分布直方图')
plt.xlabel('价格 (元)')
plt.ylabel('频数')

# 价格分布箱线图
plt.subplot(1, 3, 2)
plt.boxplot(df_train['price'])
plt.title('价格分布箱线图')
plt.ylabel('价格 (元)')

# 价格对数分布
plt.subplot(1, 3, 3)
plt.hist(np.log1p(df_train['price']), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('价格对数分布')
plt.xlabel('log(价格+1)')
plt.ylabel('频数')

plt.tight_layout()
plt.savefig('price_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 数值型特征分析
print("\n=== 4. 数值型特征分析 ===")
numeric_cols = ['power', 'kilometer', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox']
numeric_cols = [col for col in numeric_cols if col in df_train.columns]

plt.figure(figsize=(20, 15))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    plt.hist(df_train[col].dropna(), bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title(f'{col} 分布')
    plt.xlabel(col)
    plt.ylabel('频数')

plt.tight_layout()
plt.savefig('numeric_features_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 分类特征分析
print("\n=== 5. 分类特征分析 ===")
categorical_cols = ['brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'seller', 'offerType']

plt.figure(figsize=(20, 15))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 3, i)
    value_counts = df_train[col].value_counts().head(10)
    plt.bar(range(len(value_counts)), value_counts.values)
    plt.title(f'{col} 分布 (Top 10)')
    plt.xlabel(col)
    plt.ylabel('频数')
    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)

plt.tight_layout()
plt.savefig('categorical_features_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 时间特征分析
print("\n=== 6. 时间特征分析 ===")
# 转换时间格式，处理无效日期
def safe_convert_date(date_str):
    try:
        return pd.to_datetime(str(date_str), format='%Y%m%d')
    except:
        return pd.NaT

df_train['regDate'] = df_train['regDate'].apply(safe_convert_date)
df_train['creatDate'] = df_train['creatDate'].apply(safe_convert_date)

# 计算车龄
df_train['car_age'] = (df_train['creatDate'] - df_train['regDate']).dt.days / 365.25

plt.figure(figsize=(15, 5))

# 车龄分布
plt.subplot(1, 3, 1)
valid_age = df_train['car_age'].dropna()
plt.hist(valid_age, bins=30, alpha=0.7, color='purple', edgecolor='black')
plt.title('车龄分布')
plt.xlabel('车龄 (年)')
plt.ylabel('频数')

# 车龄与价格关系
plt.subplot(1, 3, 2)
valid_data = df_train[['car_age', 'price']].dropna()
plt.scatter(valid_data['car_age'], valid_data['price'], alpha=0.5, s=1)
plt.title('车龄与价格关系')
plt.xlabel('车龄 (年)')
plt.ylabel('价格 (元)')

# 上牌年份分布
plt.subplot(1, 3, 3)
df_train['regYear'] = df_train['regDate'].dt.year
year_counts = df_train['regYear'].value_counts().sort_index()
plt.bar(year_counts.index, year_counts.values, alpha=0.7, color='brown')
plt.title('上牌年份分布')
plt.xlabel('年份')
plt.ylabel('频数')

plt.tight_layout()
plt.savefig('time_features_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 相关性分析
print("\n=== 7. 相关性分析 ===")
# 选择数值型特征进行相关性分析
correlation_cols = ['price', 'power', 'kilometer', 'model', 'brand'] + [f'v_{i}' for i in range(15)]
correlation_cols = [col for col in correlation_cols if col in df_train.columns]

correlation_matrix = df_train[correlation_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 价格与其他特征的关系
print("\n=== 8. 价格与其他特征的关系 ===")
plt.figure(figsize=(20, 15))

# 功率与价格
plt.subplot(2, 3, 1)
plt.scatter(df_train['power'], df_train['price'], alpha=0.5, s=1)
plt.title('功率与价格关系')
plt.xlabel('功率 (hp)')
plt.ylabel('价格 (元)')

# 公里数与价格
plt.subplot(2, 3, 2)
plt.scatter(df_train['kilometer'], df_train['price'], alpha=0.5, s=1)
plt.title('公里数与价格关系')
plt.xlabel('公里数 (万公里)')
plt.ylabel('价格 (元)')

# 品牌与价格
plt.subplot(2, 3, 3)
brand_price = df_train.groupby('brand')['price'].mean().sort_values(ascending=False)
plt.bar(range(len(brand_price)), brand_price.values)
plt.title('各品牌平均价格')
plt.xlabel('品牌编码')
plt.ylabel('平均价格 (元)')
plt.xticks(range(len(brand_price)), brand_price.index)

# 车身类型与价格
plt.subplot(2, 3, 4)
bodytype_price = df_train.groupby('bodyType')['price'].mean().sort_values(ascending=False)
plt.bar(range(len(bodytype_price)), bodytype_price.values)
plt.title('各车身类型平均价格')
plt.xlabel('车身类型编码')
plt.ylabel('平均价格 (元)')
plt.xticks(range(len(bodytype_price)), bodytype_price.index)

# 变速箱与价格
plt.subplot(2, 3, 5)
gearbox_price = df_train.groupby('gearbox')['price'].mean()
plt.bar(gearbox_price.index, gearbox_price.values)
plt.title('变速箱类型平均价格')
plt.xlabel('变速箱类型 (0=手动, 1=自动)')
plt.ylabel('平均价格 (元)')

# 损坏情况与价格
plt.subplot(2, 3, 6)
damage_price = df_train.groupby('notRepairedDamage')['price'].mean()
plt.bar(range(len(damage_price)), damage_price.values)
plt.title('损坏情况平均价格')
plt.xlabel('损坏情况 (0=无, 1=有, -=未知)')
plt.ylabel('平均价格 (元)')
plt.xticks(range(len(damage_price)), damage_price.index)

plt.tight_layout()
plt.savefig('price_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 匿名特征分析
print("\n=== 9. 匿名特征(v_0~v_14)分析 ===")
v_cols = [f'v_{i}' for i in range(15)]

plt.figure(figsize=(20, 10))
for i, col in enumerate(v_cols, 1):
    plt.subplot(3, 5, i)
    plt.hist(df_train[col], bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title(f'{col} 分布')
    plt.xlabel(col)

plt.tight_layout()
plt.savefig('anonymous_features_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 数据质量报告
print("\n=== 10. 数据质量报告 ===")
print("训练集数据质量:")
print(f"总样本数: {len(df_train)}")
print(f"总特征数: {len(df_train.columns)}")
print(f"缺失值总数: {df_train.isnull().sum().sum()}")
print(f"缺失值比例: {df_train.isnull().sum().sum() / (len(df_train) * len(df_train.columns)):.2%}")

print("\n各字段缺失值情况:")
for col in df_train.columns:
    missing_count = df_train[col].isnull().sum()
    if missing_count > 0:
        missing_pct = missing_count / len(df_train) * 100
        print(f"{col}: {missing_count} ({missing_pct:.2f}%)")

print("\n价格统计:")
print(f"价格范围: {df_train['price'].min():,.0f} - {df_train['price'].max():,.0f}")
print(f"平均价格: {df_train['price'].mean():,.0f}")
print(f"价格中位数: {df_train['price'].median():,.0f}")
print(f"价格标准差: {df_train['price'].std():,.0f}")

print("\nEDA分析完成！所有图表已保存。") 