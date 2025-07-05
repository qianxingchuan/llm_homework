import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# 设置随机种子
np.random.seed(42)

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")
    
    # 读取数据
    train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    # 分离特征和目标变量
    target = train_df['price']
    train_features = train_df.drop(['price', 'SaleID'], axis=1)
    test_features = test_df.drop(['SaleID'], axis=1)
    
    # 合并数据集进行一致的预处理
    all_data = pd.concat([train_features, test_features], ignore_index=True)
    
    return train_features, test_features, target, all_data

def robust_feature_engineering(all_data):
    """稳健的特征工程"""
    print("正在进行特征工程...")
    
    # 1. 处理缺失值 - 使用更保守的方法
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    # 数值型特征：用中位数填充
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            median_val = all_data[col].median()
            all_data[col].fillna(median_val, inplace=True)
    
    # 分类特征：用众数填充
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            mode_val = all_data[col].mode()[0]
            all_data[col].fillna(mode_val, inplace=True)
    
    # 2. 处理异常值 - 使用IQR方法
    for col in numeric_cols:
        Q1 = all_data[col].quantile(0.25)
        Q3 = all_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 将异常值限制在合理范围内
        all_data[col] = all_data[col].clip(lower_bound, upper_bound)
    
    # 3. 编码分类特征
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
        label_encoders[col] = le
    
    # 4. 添加简单的交互特征
    if 'power' in all_data.columns and 'kilometer' in all_data.columns:
        all_data['power_per_km'] = all_data['power'] / (all_data['kilometer'] + 1)
    
    if 'model' in all_data.columns and 'brand' in all_data.columns:
        all_data['model_brand'] = all_data['model'] * all_data['brand']
    
    # 5. 时间特征处理
    if 'regDate' in all_data.columns:
        # 安全转换日期
        def safe_convert_date(date_str):
            try:
                return pd.to_datetime(str(date_str), format='%Y%m%d')
            except:
                return pd.NaT
        
        all_data['regDate'] = all_data['regDate'].apply(safe_convert_date)
        all_data['regYear'] = all_data['regDate'].dt.year
        all_data['regMonth'] = all_data['regDate'].dt.month
        all_data['regYear'].fillna(all_data['regYear'].median(), inplace=True)
        all_data['regMonth'].fillna(all_data['regMonth'].median(), inplace=True)
    
    if 'creatDate' in all_data.columns:
        all_data['creatDate'] = all_data['creatDate'].apply(safe_convert_date)
        all_data['creatYear'] = all_data['creatDate'].dt.year
        all_data['creatMonth'] = all_data['creatDate'].dt.month
        all_data['creatYear'].fillna(all_data['creatYear'].median(), inplace=True)
        all_data['creatMonth'].fillna(all_data['creatMonth'].median(), inplace=True)
    
    # 6. 计算车龄
    if 'regYear' in all_data.columns and 'creatYear' in all_data.columns:
        all_data['car_age'] = all_data['creatYear'] - all_data['regYear']
        all_data['car_age'] = all_data['car_age'].clip(0, 30)  # 限制车龄在0-30年
    
    # 7. 标准化数值特征
    scaler = StandardScaler()
    numeric_features = all_data.select_dtypes(include=[np.number]).columns
    all_data[numeric_features] = scaler.fit_transform(all_data[numeric_features])
    
    return all_data, label_encoders, scaler

def train_robust_models(X_train, y_train):
    """训练稳健的模型集成"""
    print("正在训练模型...")
    
    # 1. 基础模型
    models = {
        'xgb': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'lgb': lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'rf': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'gbm': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'ridge': Ridge(alpha=1.0, random_state=42)
    }
    
    # 2. 交叉验证评估
    cv_scores = {}
    trained_models = {}
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"训练 {name}...")
        scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        cv_scores[name] = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std()
        }
        print(f"{name} - 平均RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
        
        # 训练完整模型
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # 3. 选择最佳模型
    best_model_name = min(cv_scores.keys(), key=lambda x: cv_scores[x]['mean_rmse'])
    print(f"\n最佳模型: {best_model_name}")
    
    return trained_models, cv_scores, best_model_name

def ensemble_predictions(models, X_test, best_model_name):
    """集成预测"""
    print("正在进行集成预测...")
    
    predictions = {}
    
    # 获取各个模型的预测
    for name, model in models.items():
        pred = model.predict(X_test)
        predictions[name] = pred
    
    # 加权集成（给最佳模型更高权重）
    weights = {
        'xgb': 0.4,
        'lgb': 0.3,
        'rf': 0.15,
        'gbm': 0.1,
        'ridge': 0.05
    }
    
    # 确保权重和为1
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # 加权平均
    ensemble_pred = np.zeros(len(X_test))
    for name, weight in weights.items():
        ensemble_pred += weight * predictions[name]
    
    return ensemble_pred, predictions

def post_process_predictions(predictions, target_stats):
    """预测后处理"""
    print("正在进行预测后处理...")
    
    # 1. 获取目标变量的统计信息
    target_mean = target_stats['mean']
    target_std = target_stats['std']
    target_min = target_stats['min']
    target_max = target_stats['max']
    
    # 2. 限制预测范围在合理区间内
    # 使用训练集的统计信息来限制预测范围
    lower_bound = max(0, target_min - 2 * target_std)  # 不低于0
    upper_bound = target_max + 2 * target_std
    
    processed_predictions = np.clip(predictions, lower_bound, upper_bound)
    
    # 3. 平滑处理 - 使用移动平均
    window_size = 5
    if len(processed_predictions) > window_size:
        smoothed = np.convolve(processed_predictions, np.ones(window_size)/window_size, mode='same')
        # 只对中间部分使用平滑，保持首尾不变
        processed_predictions[window_size:-window_size] = smoothed[window_size:-window_size]
    
    return processed_predictions

def main():
    """主函数"""
    print("=== 稳健优化策略 ===")
    
    # 1. 加载和预处理数据
    train_features, test_features, target, all_data = load_and_preprocess_data()
    
    # 2. 特征工程
    processed_data, label_encoders, scaler = robust_feature_engineering(all_data)
    
    # 3. 分离训练集和测试集
    train_size = len(train_features)
    X_train = processed_data[:train_size]
    X_test = processed_data[train_size:]
    
    # 4. 获取目标变量统计信息
    target_stats = {
        'mean': target.mean(),
        'std': target.std(),
        'min': target.min(),
        'max': target.max(),
        'median': target.median()
    }
    
    print(f"目标变量统计: {target_stats}")
    
    # 5. 训练模型
    models, cv_scores, best_model = train_robust_models(X_train, target)
    
    # 6. 集成预测
    ensemble_pred, individual_preds = ensemble_predictions(models, X_test, best_model)
    
    # 7. 后处理
    final_predictions = post_process_predictions(ensemble_pred, target_stats)
    
    # 8. 保存结果
    submission = pd.DataFrame({
        'SaleID': range(200000, 200000 + len(final_predictions)),
        'price': final_predictions
    })
    
    submission.to_csv('robust_optimization_submission.csv', index=False)
    print(f"预测结果已保存到 robust_optimization_submission.csv")
    
    # 9. 输出预测统计信息
    print(f"\n预测结果统计:")
    print(f"预测值范围: {final_predictions.min():.2f} - {final_predictions.max():.2f}")
    print(f"预测值均值: {final_predictions.mean():.2f}")
    print(f"预测值中位数: {np.median(final_predictions):.2f}")
    
    # 10. 保存模型性能报告
    with open('robust_optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== 稳健优化策略报告 ===\n\n")
        f.write("模型交叉验证结果:\n")
        for name, scores in cv_scores.items():
            f.write(f"{name}: RMSE = {scores['mean_rmse']:.2f} ± {scores['std_rmse']:.2f}\n")
        f.write(f"\n最佳模型: {best_model}\n")
        f.write(f"\n目标变量统计: {target_stats}\n")
        f.write(f"\n预测结果统计:\n")
        f.write(f"预测值范围: {final_predictions.min():.2f} - {final_predictions.max():.2f}\n")
        f.write(f"预测值均值: {final_predictions.mean():.2f}\n")
        f.write(f"预测值中位数: {np.median(final_predictions):.2f}\n")
    
    print("优化完成！")

if __name__ == "__main__":
    main() 