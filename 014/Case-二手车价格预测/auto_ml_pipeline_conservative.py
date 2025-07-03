import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=== 二手车价格预测 - 保守版AutoML ===")

# 1. 数据加载
df_train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
df_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 2. 保守的数据预处理
def preprocess_data_conservative(df, is_train=True):
    df_processed = df.copy()
    
    # 基础日期处理
    def safe_convert_date(date_str):
        try:
            return pd.to_datetime(str(date_str), format='%Y%m%d')
        except:
            return pd.NaT
    
    df_processed['regDate'] = df_processed['regDate'].apply(safe_convert_date)
    df_processed['creatDate'] = df_processed['creatDate'].apply(safe_convert_date)
    df_processed['car_age'] = (df_processed['creatDate'] - df_processed['regDate']).dt.days / 365.25
    
    # 缺失值处理
    if 'notRepairedDamage' in df_processed.columns:
        df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].replace('-', 2)
        df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].astype(float).fillna(0).astype(int)
    
    if 'brand' in df_processed.columns:
        brand_mode = df_processed['brand'].mode()[0]
        df_processed['brand'] = df_processed['brand'].fillna(brand_mode)
    
    # 数值型特征用中位数填充
    numeric_cols = ['car_age', 'kilometer', 'model', 'bodyType', 'fuelType', 'gearbox']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # 只保留稳定的交互特征
    if all(col in df_processed.columns for col in ['power', 'kilometer']):
        df_processed['power_density'] = df_processed['power'] / (df_processed['kilometer'] + 1)
    
    if all(col in df_processed.columns for col in ['power', 'car_age']):
        df_processed['power_age_ratio'] = df_processed['power'] / (df_processed['car_age'] + 1)
    
    # 基础特征列表（去掉可能有问题的特征）
    base_features = ['power', 'kilometer', 'model', 'brand', 'bodyType', 'fuelType', 
                    'gearbox', 'notRepairedDamage', 'regionCode', 'car_age']
    
    # 稳定的交互特征
    interaction_features = ['power_density', 'power_age_ratio']
    
    # v特征
    v_features = [f'v_{i}' for i in range(15)]
    
    feature_cols = (base_features + 
                   [f for f in interaction_features if f in df_processed.columns] +
                   v_features)
    
    available_cols = [col for col in feature_cols if col in df_processed.columns]
    
    if is_train:
        return df_processed[available_cols], df_processed['price']
    else:
        return df_processed[available_cols]

print("2. 保守的数据预处理...")
X, y = preprocess_data_conservative(df_train, is_train=True)

# 3. 特征选择（更保守，只选Top 20）
print("3. 保守特征选择...")
selector_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
selector_model.fit(X, y)
importances = selector_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("Top 20特征:")
print(feat_imp.head(20))
selected_features = feat_imp.head(20).index.tolist()
X_selected = X[selected_features]

# 4. 数据分割
print("4. 数据分割...")
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 5. 保守的参数调优
print("5. 保守参数调优...")
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}
gs = GridSearchCV(xgb.XGBRegressor(random_state=42, n_jobs=-1), param_grid, cv=3, 
                 scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print(f"最佳参数: {gs.best_params_}")

# 6. 训练最终模型
print("6. 训练最终模型...")
final_model = xgb.XGBRegressor(**gs.best_params_, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

# 7. 验证集评估
print("7. 验证集评估...")
y_pred_val = final_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
print(f"验证集RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

# 8. 预测测试集
print("8. 预测测试集并保存结果...")
X_test = preprocess_data_conservative(df_test, is_train=False)[selected_features]
y_test_pred = final_model.predict(X_test)

# 检查预测结果范围
print(f"预测值范围: {y_test_pred.min():.2f} - {y_test_pred.max():.2f}")
print(f"预测值中位数: {np.median(y_test_pred):.2f}")
print(f"预测值均值: {np.mean(y_test_pred):.2f}")

# 保存结果
submission = pd.DataFrame({'SaleID': df_test['SaleID'], 'price': y_test_pred})
submission.to_csv('auto_ml_conservative_submission.csv', index=False)
print("提交文件已保存: auto_ml_conservative_submission.csv")

print("\n=== 保守版AutoML完成！===") 