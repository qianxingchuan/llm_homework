import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=== 二手车价格预测 - 高级优化策略 ===")

# 1. 数据加载
print("\n1. 数据加载...")
df_train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
df_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集形状: {df_train.shape}")
print(f"测试集形状: {df_test.shape}")

# 2. 分析目标变量分布
print("\n2. 分析目标变量分布...")
print("价格统计信息:")
print(df_train['price'].describe())

# 检查价格分布
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df_train['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('原始价格分布')
plt.xlabel('价格 (元)')
plt.ylabel('频数')

plt.subplot(1, 3, 2)
plt.hist(np.log1p(df_train['price']), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('对数价格分布')
plt.xlabel('log(价格+1)')
plt.ylabel('频数')

plt.subplot(1, 3, 3)
plt.boxplot(df_train['price'])
plt.title('价格箱线图')
plt.ylabel('价格 (元)')

plt.tight_layout()
plt.savefig('price_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 高级数据预处理
print("\n3. 高级数据预处理...")

def advanced_preprocess_data(df, is_train=True):
    """高级数据预处理函数"""
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
    
    # 智能填充缺失值
    # 对于数值型特征，使用中位数填充
    numeric_cols = ['model', 'bodyType', 'fuelType', 'gearbox', 'car_age']
    for col in numeric_cols:
        if col in df_processed.columns:
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
    
    # 处理notRepairedDamage
    df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].replace('-', 2)
    df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].astype(float).astype(int)
    
    # 高级特征工程
    # 功率特征
    df_processed['power_bin'] = pd.cut(df_processed['power'], 
                                     bins=[0, 100, 150, 200, 300, 1000], 
                                     labels=[0, 1, 2, 3, 4], 
                                     include_lowest=True)
    df_processed['power_bin'] = df_processed['power_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
    
    # 公里数特征
    df_processed['kilometer_bin'] = pd.cut(df_processed['kilometer'], 
                                         bins=[0, 5, 10, 15, 20, 50], 
                                         labels=[0, 1, 2, 3, 4], 
                                         include_lowest=True)
    df_processed['kilometer_bin'] = df_processed['kilometer_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
    
    # 车龄特征
    df_processed['car_age_bin'] = pd.cut(df_processed['car_age'], 
                                       bins=[0, 3, 6, 10, 15, 50], 
                                       labels=[0, 1, 2, 3, 4], 
                                       include_lowest=True)
    df_processed['car_age_bin'] = df_processed['car_age_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
    
    # 新增特征：功率密度（功率/车龄）
    df_processed['power_density'] = df_processed['power'] / (df_processed['car_age'] + 1)
    
    # 新增特征：公里数密度（公里数/车龄）
    df_processed['kilometer_density'] = df_processed['kilometer'] / (df_processed['car_age'] + 1)
    
    # 新增特征：品牌车型组合
    df_processed['brand_model'] = df_processed['brand'].astype(str) + '_' + df_processed['model'].astype(str)
    
    # 选择特征
    feature_cols = ['power', 'kilometer', 'model', 'brand', 'bodyType', 'fuelType', 
                   'gearbox', 'notRepairedDamage', 'regionCode', 'car_age',
                   'power_bin', 'kilometer_bin', 'car_age_bin', 'power_density', 'kilometer_density'] + [f'v_{i}' for i in range(15)]
    
    available_cols = [col for col in feature_cols if col in df_processed.columns]
    
    if is_train:
        return df_processed[available_cols], df_processed['price']
    else:
        return df_processed[available_cols]

# 预处理数据
X_train, y_train = advanced_preprocess_data(df_train, is_train=True)
X_test = advanced_preprocess_data(df_test, is_train=False)

print(f"训练特征形状: {X_train.shape}")
print(f"测试特征形状: {X_test.shape}")

# 4. 目标变量变换策略
print("\n4. 目标变量变换策略...")

# 策略1：原始价格
y_train_original = y_train.copy()

# 策略2：对数变换
y_train_log = np.log1p(y_train)

# 策略3：平方根变换
y_train_sqrt = np.sqrt(y_train)

# 策略4：Box-Cox变换（近似）
y_train_boxcox = np.log1p(y_train - y_train.min() + 1)

# 5. 模型训练和比较
print("\n5. 模型训练和比较...")

def train_and_evaluate(X_train, y_train, strategy_name):
    """训练和评估模型"""
    print(f"\n训练策略: {strategy_name}")
    
    # 数据分割
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 基础模型
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_split, y_train_split)
    y_val_pred = model.predict(X_val_split)
    
    # 如果是对数变换，需要转换回原始尺度
    if 'log' in strategy_name:
        y_val_pred = np.expm1(y_val_pred)
        y_val_split = np.expm1(y_val_split)
    elif 'sqrt' in strategy_name:
        y_val_pred = y_val_pred ** 2
        y_val_split = y_val_split ** 2
    elif 'boxcox' in strategy_name:
        y_val_pred = np.expm1(y_val_pred) + y_train.min() - 1
        y_val_split = np.expm1(y_val_split) + y_train.min() - 1
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
    mae = mean_absolute_error(y_val_split, y_val_pred)
    r2 = r2_score(y_val_split, y_val_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return model, rmse, mae, r2

# 比较不同策略
strategies = [
    ('原始价格', y_train_original),
    ('对数变换', y_train_log),
    ('平方根变换', y_train_sqrt),
    ('Box-Cox变换', y_train_boxcox)
]

results = []
for strategy_name, y_train_strategy in strategies:
    model, rmse, mae, r2 = train_and_evaluate(X_train, y_train_strategy, strategy_name)
    results.append({
        'strategy': strategy_name,
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })

# 6. 选择最佳策略
print("\n6. 选择最佳策略...")
best_result = min(results, key=lambda x: x['rmse'])
print(f"最佳策略: {best_result['strategy']}")
print(f"最佳RMSE: {best_result['rmse']:.2f}")
print(f"最佳MAE: {best_result['mae']:.2f}")
print(f"最佳R²: {best_result['r2']:.4f}")

# 7. 使用最佳策略进行最终预测
print("\n7. 使用最佳策略进行最终预测...")

# 重新训练最佳模型
best_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# 根据最佳策略选择目标变量
if 'log' in best_result['strategy']:
    y_train_final = y_train_log
elif 'sqrt' in best_result['strategy']:
    y_train_final = y_train_sqrt
elif 'boxcox' in best_result['strategy']:
    y_train_final = y_train_boxcox
else:
    y_train_final = y_train_original

# 训练最终模型
best_model.fit(X_train, y_train_final)

# 预测
y_test_pred = best_model.predict(X_test)

# 转换回原始尺度
if 'log' in best_result['strategy']:
    y_test_pred = np.expm1(y_test_pred)
elif 'sqrt' in best_result['strategy']:
    y_test_pred = y_test_pred ** 2
elif 'boxcox' in best_result['strategy']:
    y_test_pred = np.expm1(y_test_pred) + y_train.min() - 1

# 确保预测价格为正数
y_test_pred = np.maximum(y_test_pred, 0)

# 8. 生成提交文件
print("\n8. 生成提交文件...")
submission = pd.DataFrame({
    'SaleID': df_test['SaleID'],
    'price': y_test_pred
})

submission.to_csv('xgboost_submission_optimized.csv', index=False)
print("提交文件已保存: xgboost_submission_optimized.csv")

# 9. 保存最佳模型
print("\n9. 保存最佳模型...")
import joblib
joblib.dump(best_model, 'xgboost_model_optimized.pkl')
print("模型已保存: xgboost_model_optimized.pkl")

# 10. 预测结果分析
print("\n10. 预测结果分析...")
print(f"预测价格范围: {y_test_pred.min():.2f} - {y_test_pred.max():.2f}")
print(f"预测价格均值: {y_test_pred.mean():.2f}")
print(f"预测价格中位数: {np.median(y_test_pred):.2f}")
print(f"预测价格标准差: {y_test_pred.std():.2f}")

print("\n=== 优化总结 ===")
print(f"最佳策略: {best_result['strategy']}")
print(f"验证集RMSE: {best_result['rmse']:.2f}")
print(f"验证集MAE: {best_result['mae']:.2f}")
print(f"验证集R²: {best_result['r2']:.4f}")
print(f"预测记录数: {len(submission)}")

print("\n优化完成！") 