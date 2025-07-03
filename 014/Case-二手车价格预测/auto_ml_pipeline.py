import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=== 二手车价格预测 - AutoML全流程脚本 ===")

# 1. 数据加载
df_train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
df_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

# 2. 数据预处理和特征工程
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
    # 多项式特征（只对数值型主特征）
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_cols = ['power', 'kilometer', 'car_age']
    for col in poly_cols:
        if col not in df_processed.columns:
            poly_cols.remove(col)
    if len(poly_cols) >= 2:
        poly_features = poly.fit_transform(df_processed[poly_cols])
        poly_feature_names = poly.get_feature_names_out(poly_cols)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_processed.index)
        # 只保留交互项（去掉原始特征）
        poly_df = poly_df[[c for c in poly_df.columns if '*' in c]]
        df_processed = pd.concat([df_processed, poly_df], axis=1)
        available_cols += list(poly_df.columns)
    if is_train:
        return df_processed[available_cols], df_processed['price']
    else:
        return df_processed[available_cols]

print("2. 数据预处理和特征工程...")
X, y = preprocess_data(df_train, is_train=True)

# 3. 自动特征选择（基于XGBoost特征重要性，选Top 30）
print("3. 自动特征选择...")
selector_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
selector_model.fit(X, y)
importances = selector_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("Top 30特征:")
print(feat_imp.head(30))
selected_features = feat_imp.head(30).index.tolist()
X_selected = X[selected_features]

# 4. 数据分割
print("4. 数据分割...")
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 5. 参数调优（XGBoost为例）
print("5. XGBoost参数调优...")
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.7, 0.8]
}
gs = GridSearchCV(xgb.XGBRegressor(random_state=42, n_jobs=-1), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print(f"最佳参数: {gs.best_params_}")

# 6. 模型融合（Stacking）
print("6. 模型融合（Stacking）...")
base_models = [
    ('xgb', xgb.XGBRegressor(**gs.best_params_, random_state=42, n_jobs=-1)),
    ('lgb', lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=-1)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1))
]

# 单独训练每个基础模型
print("训练基础模型...")
trained_models = []
for name, model in base_models:
    print(f"训练 {name}...")
    model.fit(X_train, y_train)
    trained_models.append((name, model))

# 训练Stacking模型
print("训练Stacking融合模型...")
stack = StackingRegressor(estimators=base_models, final_estimator=Ridge(), n_jobs=-1)
stack.fit(X_train, y_train)

# 7. 验证集评估
print("7. 验证集评估...")
def eval_model(name, model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"{name} 验证集RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    return rmse, mae, r2

# 单模型评估
for name, model in trained_models:
    eval_model(name, model, X_val, y_val)

# 融合模型评估
eval_model('Stacking融合', stack, X_val, y_val)

# 8. 用融合模型预测测试集并保存
print("8. 预测测试集并保存结果...")
X_test = preprocess_data(df_test, is_train=False)[selected_features]
y_test_pred = stack.predict(X_test)
submission = pd.DataFrame({'SaleID': df_test['SaleID'], 'price': y_test_pred})
submission.to_csv('auto_ml_submission.csv', index=False)
print("提交文件已保存: auto_ml_submission.csv")

print("\n=== 全流程AutoML完成！===") 