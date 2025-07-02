import pandas as pd
import numpy as np

# 读取训练数据（包含价格信息）
print("=== 分析训练数据 ===")
df_train = pd.read_csv('used_car_train_20200313.csv', sep=' ')

print("训练数据集基本信息:")
print(f"数据形状: {df_train.shape}")
print(f"列名: {list(df_train.columns)}")
print("\n前3行数据:")
print(df_train.head(3))

print("\n数据类型:")
print(df_train.dtypes)

print("\n缺失值统计:")
print(df_train.isnull().sum())

print("\n数值型字段的基本统计:")
numeric_cols = df_train.select_dtypes(include=[np.number]).columns
print(df_train[numeric_cols].describe())

print("\n分类字段的唯一值:")
categorical_cols = ['name', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode', 'seller', 'offerType']
for col in categorical_cols:
    if col in df_train.columns:
        print(f"\n{col}:")
        print(df_train[col].value_counts().head(10))

print("\n=== 分析测试数据 ===")
df_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
print(f"测试数据形状: {df_test.shape}")
print(f"测试数据列名: {list(df_test.columns)}") 