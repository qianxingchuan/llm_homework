import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    has_lgb = True
except ImportError:
    has_lgb = False

print("=== 二手车价格预测 - 多模型对比 ===")

# 1. 数据加载
df_train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
df_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 2. 数据预处理（与weighted_feature_model.py一致）
def preprocess_data(df, is_train=True):
    df_processed = df.copy()
    def safe_convert_date(date_str):
        try:
            return pd.to_datetime(str(date_str), format='%Y%m%d')
        except:
            return pd.NaT
    df_processed['regDate'] = df_processed['regDate'].apply(safe_convert_date)
    df_processed['creatDate'] = df_processed['creatDate'].apply(safe_convert_date)
    df_processed['car_age'] = (df_processed['creatDate'] - df_processed['regDate']).dt.days / 365.25
    if 'notRepairedDamage' in df_processed.columns:
        df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].replace('-', 2)
        df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].astype(float).fillna(0).astype(int)
    if 'brand' in df_processed.columns:
        brand_mode = df_processed['brand'].mode()[0]
        df_processed['brand'] = df_processed['brand'].fillna(brand_mode)
    if 'car_age' in df_processed.columns:
        df_processed['car_age'] = df_processed['car_age'].fillna(df_processed['car_age'].median())
    if 'kilometer' in df_processed.columns:
        df_processed['kilometer'] = df_processed['kilometer'].fillna(df_processed['kilometer'].median())
    numeric_cols = ['model', 'bodyType', 'fuelType', 'gearbox']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    # 交互特征
    if all(col in df_processed.columns for col in ['notRepairedDamage', 'brand']):
        df_processed['damage_brand'] = df_processed['notRepairedDamage'] * df_processed['brand']
    if all(col in df_processed.columns for col in ['notRepairedDamage', 'car_age']):
        df_processed['damage_age'] = df_processed['notRepairedDamage'] * df_processed['car_age']
    if all(col in df_processed.columns for col in ['brand', 'car_age']):
        df_processed['brand_age'] = df_processed['brand'] * df_processed['car_age']
    if all(col in df_processed.columns for col in ['kilometer', 'car_age']):
        df_processed['km_age'] = df_processed['kilometer'] * df_processed['car_age']
    if all(col in df_processed.columns for col in ['power', 'kilometer']):
        df_processed['power_density'] = df_processed['power'] / (df_processed['kilometer'] + 1)
    if all(col in df_processed.columns for col in ['power', 'car_age']):
        df_processed['power_age_ratio'] = df_processed['power'] / (df_processed['car_age'] + 1)
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
    base_features = ['power', 'kilometer', 'model', 'brand', 'bodyType', 'fuelType', 
                    'gearbox', 'notRepairedDamage', 'regionCode', 'car_age']
    interaction_features = ['damage_brand', 'damage_age', 'brand_age', 'km_age', 
                           'power_density', 'power_age_ratio']
    bin_features = ['power_bin', 'kilometer_bin', 'car_age_bin']
    v_features = [f'v_{i}' for i in range(15)]
    feature_cols = (base_features + 
                   [f for f in interaction_features if f in df_processed.columns] +
                   bin_features + 
                   v_features)
    available_cols = [col for col in feature_cols if col in df_processed.columns]
    if is_train:
        return df_processed[available_cols], df_processed['price']
    else:
        return df_processed[available_cols]

X, y = preprocess_data(df_train, is_train=True)

# 3. 数据分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 定义模型
models = {
    'XGBoost': xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=-1),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1)
}
if has_lgb:
    models['LightGBM'] = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=-1)

# 5. 训练与评估
results = {}
for name, model in models.items():
    print(f"\n训练模型: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"{name} 验证集RMSE: {rmse:.2f}")
    print(f"{name} 验证集MAE: {mae:.2f}")
    print(f"{name} 验证集R²: {r2:.4f}")

# 6. 总结对比
print("\n=== 模型对比结果 ===")
print("模型\t\tRMSE\t\tMAE\t\tR²")
for name, res in results.items():
    print(f"{name}\t{res['RMSE']:.2f}\t{res['MAE']:.2f}\t{res['R2']:.4f}")

print("\n对比完成！") 