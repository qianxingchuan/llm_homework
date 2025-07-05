import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

def simulate_platform_mae():
    """模拟平台MAE计算方式"""
    print("=== 模拟平台MAE计算 ===")
    
    # 加载数据
    train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    print(f"训练数据形状: {train_data.shape}")
    
    # 先做异常值过滤
    train_data = train_data[train_data['power'] < 1000]
    train_data = train_data[train_data['kilometer'] <= 20]
    train_data = train_data[(train_data['price'] > 1) & (train_data['price'] < 2e5)]
    print(f"过滤后训练数据形状: {train_data.shape}")
    
    # 划分训练集和验证集（模拟测试集）
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    print(f"训练集: {train_data.shape}, 验证集: {val_data.shape}")
    
    # 保存真实价格用于计算MAE
    true_prices = val_data['price'].values
    
    # 特征工程（简化版，只做基础处理）
    def basic_feature_engineering(data):
        # 基础特征
        features = data.drop(['price', 'SaleID'], axis=1)
        
        # 处理缺失值
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        categorical_cols = features.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if features[col].isnull().sum() > 0:
                if col.startswith('v_'):
                    features[col] = features[col].fillna(0)
                else:
                    features[col] = features[col].fillna(features[col].median())
        
        for col in categorical_cols:
            if features[col].isnull().sum() > 0:
                features[col] = features[col].fillna(features[col].mode()[0])
        
        # 编码分类特征
        for col in categorical_cols:
            features[col] = pd.Categorical(features[col]).codes
        
        # 确保所有列都是数值类型
        for col in features.columns:
            if features[col].dtype == 'object':
                features[col] = pd.to_numeric(features[col], errors='coerce')
                features[col] = features[col].fillna(features[col].median())
        
        return features.select_dtypes(include=[np.number])
    
    # 处理训练集和验证集
    X_train = basic_feature_engineering(train_data)
    X_val = basic_feature_engineering(val_data)
    y_train = train_data['price'].values
    y_val = val_data['price'].values
    
    print(f"特征数量: {X_train.shape[1]}")
    print(f"训练集特征形状: {X_train.shape}, 目标形状: {y_train.shape}")
    print(f"验证集特征形状: {X_val.shape}, 目标形状: {y_val.shape}")
    
    # 训练XGBoost（原始价格空间）
    print("\n=== 训练XGBoost（原始价格空间）===")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='gpu_hist',
        gpu_id=0,
        max_depth=8,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    xgb_mae = mean_absolute_error(y_val, xgb_pred)
    print(f"XGBoost 原始价格空间MAE: {xgb_mae:.2f}")
    
    # 训练LightGBM（原始价格空间）
    print("\n=== 训练LightGBM（原始价格空间）===")
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        max_depth=8,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    print(f"LightGBM 原始价格空间MAE: {lgb_mae:.2f}")
    
    # 集成预测
    print("\n=== 集成预测（原始价格空间）===")
    # 权重基于MAE倒数
    xgb_weight = 1.0 / xgb_mae
    lgb_weight = 1.0 / lgb_mae
    total_weight = xgb_weight + lgb_weight
    
    xgb_weight /= total_weight
    lgb_weight /= total_weight
    
    ensemble_pred = xgb_weight * xgb_pred + lgb_weight * lgb_pred
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    
    print(f"XGBoost权重: {xgb_weight:.4f}")
    print(f"LightGBM权重: {lgb_weight:.4f}")
    print(f"集成MAE: {ensemble_mae:.2f}")
    
    # 对比对数变换的效果
    print("\n=== 对比对数变换效果 ===")
    # 对数变换训练
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    xgb_model_log = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='gpu_hist',
        gpu_id=0,
        max_depth=8,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    
    xgb_model_log.fit(X_train, y_train_log)
    xgb_pred_log = xgb_model_log.predict(X_val)
    xgb_pred_original = np.expm1(xgb_pred_log)
    xgb_mae_log = mean_absolute_error(y_val, xgb_pred_original)
    print(f"XGBoost 对数变换后MAE: {xgb_mae_log:.2f}")
    
    return {
        'xgb_original': xgb_mae,
        'lgb_original': lgb_mae,
        'ensemble_original': ensemble_mae,
        'xgb_log': xgb_mae_log
    }

if __name__ == "__main__":
    results = simulate_platform_mae()
    
    print("\n=== 总结 ===")
    print(f"原始价格空间:")
    print(f"  XGBoost: {results['xgb_original']:.2f}")
    print(f"  LightGBM: {results['lgb_original']:.2f}")
    print(f"  集成: {results['ensemble_original']:.2f}")
    print(f"对数变换空间:")
    print(f"  XGBoost: {results['xgb_log']:.2f}")
    
    print(f"\n对数变换 vs 原始空间:")
    print(f"  XGBoost改进: {results['xgb_original'] - results['xgb_log']:.2f}")
    print(f"  改进百分比: {(results['xgb_original'] - results['xgb_log']) / results['xgb_original'] * 100:.1f}%") 