import pandas as pd
import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_sample_data():
    """加载样本数据进行速度测试"""
    print("加载数据...")
    
    # 加载训练数据
    train_data = pd.read_csv('used_car_train_20200313.csv')
    
    # 简单预处理
    # 处理日期特征
    for col in ['regDate', 'creatDate']:
        if col in train_data.columns:
            train_data[col] = pd.to_datetime(train_data[col], errors='coerce')
            train_data[f'{col}_year'] = train_data[col].dt.year
            train_data[f'{col}_month'] = train_data[col].dt.month
            train_data = train_data.drop(columns=[col])
    
    # 处理分类特征
    categorical_columns = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    for col in categorical_columns:
        if col in train_data.columns:
            train_data[col] = train_data[col].astype('category').cat.codes
    
    # 处理数值特征
    numeric_columns = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
    for col in numeric_columns:
        if col in train_data.columns:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
            train_data[col] = train_data[col].fillna(train_data[col].median())
    
    # 目标变量对数变换
    y = np.log1p(train_data['price'])
    X = train_data.drop(columns=['price'])
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def test_xgboost_speed(X_train, y_train, X_test, y_test):
    """测试XGBoost CPU vs GPU速度"""
    print("\n=== XGBoost 速度测试 ===")
    
    # CPU版本
    print("训练 XGBoost (CPU)...")
    start_time = time.time()
    
    xgb_cpu = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42,
        verbosity=0
    )
    
    xgb_cpu.fit(X_train, y_train)
    cpu_time = time.time() - start_time
    
    cpu_pred = xgb_cpu.predict(X_test)
    cpu_rmse = np.sqrt(mean_squared_error(y_test, cpu_pred))
    
    print(f"CPU 训练时间: {cpu_time:.2f}秒")
    print(f"CPU RMSE: {cpu_rmse:.4f}")
    
    # GPU版本
    print("训练 XGBoost (GPU)...")
    start_time = time.time()
    
    xgb_gpu = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='gpu_hist',
        gpu_id=0,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42,
        verbosity=0
    )
    
    xgb_gpu.fit(X_train, y_train)
    gpu_time = time.time() - start_time
    
    gpu_pred = xgb_gpu.predict(X_test)
    gpu_rmse = np.sqrt(mean_squared_error(y_test, gpu_pred))
    
    print(f"GPU 训练时间: {gpu_time:.2f}秒")
    print(f"GPU RMSE: {gpu_rmse:.4f}")
    
    # 计算加速比
    speedup = cpu_time / gpu_time
    print(f"GPU加速比: {speedup:.2f}x")
    
    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup,
        'cpu_rmse': cpu_rmse,
        'gpu_rmse': gpu_rmse
    }

def test_lightgbm_speed(X_train, y_train, X_test, y_test):
    """测试LightGBM CPU vs GPU速度"""
    print("\n=== LightGBM 速度测试 ===")
    
    # CPU版本
    print("训练 LightGBM (CPU)...")
    start_time = time.time()
    
    lgb_cpu = lgb.LGBMRegressor(
        objective='regression',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42,
        verbose=-1
    )
    
    lgb_cpu.fit(X_train, y_train)
    cpu_time = time.time() - start_time
    
    cpu_pred = lgb_cpu.predict(X_test)
    cpu_rmse = np.sqrt(mean_squared_error(y_test, cpu_pred))
    
    print(f"CPU 训练时间: {cpu_time:.2f}秒")
    print(f"CPU RMSE: {cpu_rmse:.4f}")
    
    # GPU版本
    print("训练 LightGBM (GPU)...")
    start_time = time.time()
    
    lgb_gpu = lgb.LGBMRegressor(
        objective='regression',
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42,
        verbose=-1
    )
    
    lgb_gpu.fit(X_train, y_train)
    gpu_time = time.time() - start_time
    
    gpu_pred = lgb_gpu.predict(X_test)
    gpu_rmse = np.sqrt(mean_squared_error(y_test, gpu_pred))
    
    print(f"GPU 训练时间: {gpu_time:.2f}秒")
    print(f"GPU RMSE: {gpu_rmse:.4f}")
    
    # 计算加速比
    speedup = cpu_time / gpu_time
    print(f"GPU加速比: {speedup:.2f}x")
    
    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup,
        'cpu_rmse': cpu_rmse,
        'gpu_rmse': gpu_rmse
    }

def main():
    """主函数"""
    print("=== GPU加速效果测试 ===")
    
    # 加载数据
    X_train, X_test, y_train, y_test = load_sample_data()
    
    print(f"训练数据: {X_train.shape}")
    print(f"测试数据: {X_test.shape}")
    
    # 测试XGBoost
    xgb_results = test_xgboost_speed(X_train, y_train, X_test, y_test)
    
    # 测试LightGBM
    lgb_results = test_lightgbm_speed(X_train, y_train, X_test, y_test)
    
    # 总结
    print("\n=== 测试结果总结 ===")
    print(f"XGBoost GPU加速比: {xgb_results['speedup']:.2f}x")
    print(f"LightGBM GPU加速比: {lgb_results['speedup']:.2f}x")
    
    avg_speedup = (xgb_results['speedup'] + lgb_results['speedup']) / 2
    print(f"平均GPU加速比: {avg_speedup:.2f}x")
    
    if avg_speedup > 1.5:
        print("✅ GPU加速效果显著，建议使用GPU训练！")
    elif avg_speedup > 1.1:
        print("⚠️ GPU加速效果一般，可以考虑使用GPU")
    else:
        print("❌ GPU加速效果不明显，建议使用CPU训练")

if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error
    main() 