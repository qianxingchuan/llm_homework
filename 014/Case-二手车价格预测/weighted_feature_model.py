import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=== 二手车价格预测 - 加权特征模型 ===")

# 1. 数据加载
print("\n1. 数据加载...")
df_train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
df_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集形状: {df_train.shape}")
print(f"测试集形状: {df_test.shape}")

# 2. 特征重要性权重定义（根据用户排序）
print("\n2. 定义特征权重...")
feature_weights = {
    'notRepairedDamage': 8,  # 最重要
    'brand': 7,
    'car_age': 6,
    'kilometer': 5,
    'gearbox': 4,
    'bodyType': 3,
    'power': 2,
    'fuelType': 1  # 最不重要
}

print("特征权重排序:")
for feature, weight in sorted(feature_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {weight}")

# 3. 数据预处理
print("\n3. 数据预处理...")

def preprocess_data(df, is_train=True):
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
    
    # 处理缺失值 - 根据重要性使用不同的填充策略
    # 重要特征使用更精确的填充方法
    if 'notRepairedDamage' in df_processed.columns:
        df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].replace('-', 2)
        df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].astype(float).fillna(0).astype(int)
    
    if 'brand' in df_processed.columns:
        # 品牌使用众数填充
        brand_mode = df_processed['brand'].mode()[0]
        df_processed['brand'] = df_processed['brand'].fillna(brand_mode)
    
    if 'car_age' in df_processed.columns:
        # 车龄使用中位数填充
        df_processed['car_age'] = df_processed['car_age'].fillna(df_processed['car_age'].median())
    
    if 'kilometer' in df_processed.columns:
        # 公里数使用中位数填充
        df_processed['kilometer'] = df_processed['kilometer'].fillna(df_processed['kilometer'].median())
    
    # 其他特征使用中位数填充
    numeric_cols = ['model', 'bodyType', 'fuelType', 'gearbox']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # 4. 特征工程 - 基于权重创建交互特征
    print("4. 特征工程...")
    
    # 创建重要特征的交互项
    if all(col in df_processed.columns for col in ['notRepairedDamage', 'brand']):
        df_processed['damage_brand'] = df_processed['notRepairedDamage'] * df_processed['brand']
    
    if all(col in df_processed.columns for col in ['notRepairedDamage', 'car_age']):
        df_processed['damage_age'] = df_processed['notRepairedDamage'] * df_processed['car_age']
    
    if all(col in df_processed.columns for col in ['brand', 'car_age']):
        df_processed['brand_age'] = df_processed['brand'] * df_processed['car_age']
    
    if all(col in df_processed.columns for col in ['kilometer', 'car_age']):
        df_processed['km_age'] = df_processed['kilometer'] * df_processed['car_age']
    
    # 创建加权特征
    if all(col in df_processed.columns for col in ['power', 'kilometer']):
        df_processed['power_density'] = df_processed['power'] / (df_processed['kilometer'] + 1)
    
    if all(col in df_processed.columns for col in ['power', 'car_age']):
        df_processed['power_age_ratio'] = df_processed['power'] / (df_processed['car_age'] + 1)
    
    # 创建分类特征的分箱
    if 'power' in df_processed.columns:
        df_processed['power_bin'] = pd.cut(df_processed['power'], 
                                         bins=[0, 100, 150, 200, 300, 1000], 
                                         labels=[0, 1, 2, 3, 4], 
                                         include_lowest=True)
        df_processed['power_bin'] = df_processed['power_bin'].cat.add_categories([-1]).fillna(-1).astype(int)

    if 'kilometer' in df_processed.columns:
        df_processed['kilometer_bin'] = pd.cut(df_processed['kilometer'], 
                                             bins=[0, 5, 10, 15, 20, 50], 
                                             labels=[0, 1, 2, 3, 4], 
                                             include_lowest=True)
        df_processed['kilometer_bin'] = df_processed['kilometer_bin'].cat.add_categories([-1]).fillna(-1).astype(int)

    if 'car_age' in df_processed.columns:
        df_processed['car_age_bin'] = pd.cut(df_processed['car_age'], 
                                           bins=[0, 3, 6, 10, 15, 50], 
                                           labels=[0, 1, 2, 3, 4], 
                                           include_lowest=True)
        df_processed['car_age_bin'] = df_processed['car_age_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
    
    # 5. 特征选择 - 优先选择重要特征
    base_features = ['power', 'kilometer', 'model', 'brand', 'bodyType', 'fuelType', 
                    'gearbox', 'notRepairedDamage', 'regionCode', 'car_age']
    
    interaction_features = ['damage_brand', 'damage_age', 'brand_age', 'km_age', 
                           'power_density', 'power_age_ratio']
    
    bin_features = ['power_bin', 'kilometer_bin', 'car_age_bin']
    
    v_features = [f'v_{i}' for i in range(15)]
    
    # 按重要性排序选择特征
    feature_cols = (base_features + 
                   [f for f in interaction_features if f in df_processed.columns] +
                   bin_features + 
                   v_features)
    
    available_cols = [col for col in feature_cols if col in df_processed.columns]
    
    print(f"选择的特征数量: {len(available_cols)}")
    print("重要特征:", [col for col in available_cols if col in feature_weights.keys()])
    
    if is_train:
        return df_processed[available_cols], df_processed['price']
    else:
        return df_processed[available_cols]

# 预处理数据
X_train, y_train = preprocess_data(df_train, is_train=True)
X_test = preprocess_data(df_test, is_train=False)

print(f"训练特征形状: {X_train.shape}")
print(f"测试特征形状: {X_test.shape}")

# 6. 数据分割
print("\n5. 数据分割...")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 7. XGBoost模型训练 - 使用更适合的参数字典
print("\n6. XGBoost模型训练...")

# 基础模型
base_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

base_model.fit(X_train_split, y_train_split)
y_val_pred = base_model.predict(X_val_split)

# 评估基础模型
rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
mae = mean_absolute_error(y_val_split, y_val_pred)
r2 = r2_score(y_val_split, y_val_pred)

print(f"基础模型性能:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# 8. 超参数调优 - 针对重要特征优化
print("\n7. 超参数调优...")

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_split, y_train_split)

print(f"最佳参数: {grid_search.best_params_}")

# 使用最佳参数训练模型
best_model = grid_search.best_estimator_
best_model.fit(X_train_split, y_train_split)

y_val_pred_best = best_model.predict(X_val_split)

rmse_best = np.sqrt(mean_squared_error(y_val_split, y_val_pred_best))
mae_best = mean_absolute_error(y_val_split, y_val_pred_best)
r2_best = r2_score(y_val_split, y_val_pred_best)

print(f"最佳模型性能:")
print(f"RMSE: {rmse_best:.2f}")
print(f"MAE: {mae_best:.2f}")
print(f"R²: {r2_best:.4f}")

# 9. 特征重要性分析
print("\n8. 特征重要性分析...")

feature_importance = best_model.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 15 重要特征:")
print(importance_df.head(15))

# 检查我们关注的重要特征排名
print("\n我们关注的重要特征排名:")
for feature in feature_weights.keys():
    if feature in importance_df['feature'].values:
        rank = importance_df[importance_df['feature'] == feature].index[0] + 1
        importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
        print(f"{feature}: 排名第{rank}, 重要性{importance:.4f}")

# 10. 最终模型训练
print("\n9. 最终模型训练...")
final_model = xgb.XGBRegressor(**grid_search.best_params_, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

# 11. 预测测试集
print("\n10. 预测测试集...")
y_test_pred = final_model.predict(X_test)

# 12. 生成提交文件
print("\n11. 生成提交文件...")
submission = pd.DataFrame({
    'SaleID': df_test['SaleID'],
    'price': y_test_pred
})

submission.to_csv('xgboost_submission_weighted.csv', index=False)
print("提交文件已保存: xgboost_submission_weighted.csv")

# 13. 保存模型
print("\n12. 保存模型...")
import joblib
joblib.dump(final_model, 'xgboost_model_weighted.pkl')
print("模型已保存: xgboost_model_weighted.pkl")

print("\n=== 模型总结 ===")
print(f"训练样本数: {len(X_train)}")
print(f"测试样本数: {len(X_test)}")
print(f"特征数量: {len(X_train.columns)}")
print(f"最佳参数: {grid_search.best_params_}")
print(f"验证集RMSE: {rmse_best:.2f}")
print(f"验证集MAE: {mae_best:.2f}")
print(f"验证集R²: {r2_best:.4f}")

print("\n预测完成！") 