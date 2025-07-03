import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== 使用已训练模型进行完整预测 ===")

# 1. 加载已训练的模型
print("\n1. 加载已训练的模型...")
try:
    # 尝试加载忽略缺失值的模型
    model = joblib.load('xgboost_model_no_missing.pkl')
    print("成功加载模型: xgboost_model_no_missing.pkl")
except:
    try:
        # 尝试加载原始模型
        model = joblib.load('xgboost_model.pkl')
        print("成功加载模型: xgboost_model.pkl")
    except:
        print("错误：未找到已训练的模型文件")
        exit(1)

# 2. 加载测试数据
print("\n2. 加载测试数据...")
df_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
print(f"测试集形状: {df_test.shape}")

# 3. 数据预处理（使用填充策略）
print("\n3. 数据预处理（填充缺失值）...")

def preprocess_data_with_filling(df):
    """使用填充策略预处理数据"""
    df_processed = df.copy()
    
    # 处理时间特征
    def safe_convert_date(date_str):
        try:
            return pd.to_datetime(str(date_str), format='%Y%m%d')
        except:
            return pd.NaT
    
    df_processed['regDate'] = df_processed['regDate'].apply(safe_convert_date)
    df_processed['creatDate'] = df_processed['creatDate'].apply(safe_convert_date)
    
    # 计算车龄
    df_processed['car_age'] = (df_processed['creatDate'] - df_processed['regDate']).dt.days / 365.25
    
    # 填充缺失值
    # 数值型特征用中位数填充
    numeric_cols = ['model', 'bodyType', 'fuelType', 'gearbox', 'car_age']
    for col in numeric_cols:
        if col in df_processed.columns:
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
    
    # 处理notRepairedDamage
    df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].replace('-', 2)
    df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].astype(float).astype(int)
    
    # 特征工程
    df_processed['power_bin'] = pd.cut(df_processed['power'], 
                                     bins=[0, 100, 150, 200, 300, 1000], 
                                     labels=[0, 1, 2, 3, 4], 
                                     include_lowest=True)
    df_processed['power_bin'] = df_processed['power_bin'].cat.add_categories([-1]).fillna(-1).astype(int)

    df_processed['kilometer_bin'] = pd.cut(df_processed['kilometer'], 
                                         bins=[0, 5, 10, 15, 20, 50], 
                                         labels=[0, 1, 2, 3, 4], 
                                         include_lowest=True)
    df_processed['kilometer_bin'] = df_processed['kilometer_bin'].cat.add_categories([-1]).fillna(-1).astype(int)

    df_processed['car_age_bin'] = pd.cut(df_processed['car_age'], 
                                       bins=[0, 3, 6, 10, 15, 50], 
                                       labels=[0, 1, 2, 3, 4], 
                                       include_lowest=True)
    df_processed['car_age_bin'] = df_processed['car_age_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
    
    # 选择特征
    feature_cols = ['power', 'kilometer', 'model', 'brand', 'bodyType', 'fuelType', 
                   'gearbox', 'notRepairedDamage', 'regionCode', 'car_age',
                   'power_bin', 'kilometer_bin', 'car_age_bin'] + [f'v_{i}' for i in range(15)]
    
    available_cols = [col for col in feature_cols if col in df_processed.columns]
    
    return df_processed[available_cols]

# 预处理测试数据
X_test = preprocess_data_with_filling(df_test)
print(f"预处理后测试特征形状: {X_test.shape}")

# 4. 检查缺失值
print("\n4. 检查缺失值...")
missing_count = X_test.isnull().sum().sum()
if missing_count > 0:
    print(f"警告：仍有 {missing_count} 个缺失值")
    # 对剩余的缺失值进行填充
    for col in X_test.columns:
        if X_test[col].isnull().sum() > 0:
            if X_test[col].dtype in ['int64', 'float64']:
                X_test[col] = X_test[col].fillna(X_test[col].median())
            else:
                X_test[col] = X_test[col].fillna(X_test[col].mode()[0])
    print("已填充所有缺失值")
else:
    print("无缺失值")

# 5. 进行预测
print("\n5. 进行预测...")
y_test_pred = model.predict(X_test)
print(f"预测完成，共预测 {len(y_test_pred)} 条记录")

# 6. 生成提交文件
print("\n6. 生成提交文件...")
submission = pd.DataFrame({
    'SaleID': df_test['SaleID'],
    'price': y_test_pred
})

submission.to_csv('xgboost_submission_complete.csv', index=False)
print("提交文件已保存: xgboost_submission_complete.csv")

# 7. 预测结果统计
print("\n7. 预测结果统计...")
print(f"预测价格范围: {y_test_pred.min():.2f} - {y_test_pred.max():.2f}")
print(f"预测价格均值: {y_test_pred.mean():.2f}")
print(f"预测价格中位数: {np.median(y_test_pred):.2f}")
print(f"预测价格标准差: {y_test_pred.std():.2f}")

# 8. 检查预测结果
print("\n8. 检查预测结果...")
print("前10条预测结果:")
for i in range(min(10, len(submission))):
    print(f"SaleID: {submission.iloc[i]['SaleID']}, 预测价格: {submission.iloc[i]['price']:.2f}")

print("\n预测完成！")
print(f"总共预测了 {len(submission)} 条记录")
print("文件保存为: xgboost_submission_complete.csv") 