import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

# 设置随机种子
np.random.seed(42)

def load_and_transform_target():
    """加载数据并进行目标变量变换"""
    print("正在加载数据并进行目标变量变换...")
    
    train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    # 分析目标变量
    target = train_df['price']
    print(f"\n目标变量分析:")
    print(f"  原始均值: {target.mean():.2f}")
    print(f"  原始中位数: {target.median():.2f}")
    print(f"  原始标准差: {target.std():.2f}")
    print(f"  原始偏度: {target.skew():.2f}")
    
    # 尝试不同的变换
    log_target = np.log1p(target)
    sqrt_target = np.sqrt(target)
    
    print(f"  对数变换后偏度: {log_target.skew():.2f}")
    print(f"  平方根变换后偏度: {sqrt_target.skew():.2f}")
    
    # 选择偏度最小的变换
    if abs(log_target.skew()) < abs(sqrt_target.skew()):
        transformed_target = log_target
        transform_type = 'log'
        print("选择对数变换")
    else:
        transformed_target = sqrt_target
        transform_type = 'sqrt'
        print("选择平方根变换")
    
    return train_df, test_df, target, transformed_target, transform_type

def advanced_feature_engineering(train_df, test_df):
    """高级特征工程"""
    print("正在进行高级特征工程...")
    
    # 分离特征和目标
    train_features = train_df.drop(['price', 'SaleID'], axis=1)
    test_features = test_df.drop(['SaleID'], axis=1)
    
    # 合并数据集
    all_data = pd.concat([train_features, test_features], ignore_index=True)
    
    # 1. 处理缺失值
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    # 数值型特征
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            if col.startswith('v_'):
                all_data[col].fillna(0, inplace=True)
            else:
                median_val = all_data[col].median()
                all_data[col].fillna(median_val, inplace=True)
    
    # 分类特征
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            mode_val = all_data[col].mode()[0]
            all_data[col].fillna(mode_val, inplace=True)
    
    # 2. 异常值处理 - 更保守的方法
    for col in numeric_cols:
        if col.startswith('v_'):
            continue
        
        Q1 = all_data[col].quantile(0.25)
        Q3 = all_data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 使用更宽松的边界
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
        
        all_data[col] = all_data[col].clip(lower_bound, upper_bound)
    
    # 3. 编码分类特征
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
        label_encoders[col] = le
    
    # 4. 时间特征处理 - 确保转换为数值
    if 'regDate' in all_data.columns:
        def safe_convert_date(date_str):
            try:
                return pd.to_datetime(str(date_str), format='%Y%m%d')
            except:
                return pd.NaT
        
        all_data['regDate'] = all_data['regDate'].apply(safe_convert_date)
        all_data['regYear'] = all_data['regDate'].dt.year
        all_data['regMonth'] = all_data['regDate'].dt.month
        all_data['regDay'] = all_data['regDate'].dt.day
        all_data['regDayOfWeek'] = all_data['regDate'].dt.dayofweek
        all_data['regQuarter'] = all_data['regDate'].dt.quarter
        
        # 填充缺失值
        for col in ['regYear', 'regMonth', 'regDay', 'regDayOfWeek', 'regQuarter']:
            if col in all_data.columns:
                all_data[col].fillna(all_data[col].median(), inplace=True)
        
        # 删除原始的datetime列
        all_data.drop(['regDate'], axis=1, inplace=True)
    
    if 'creatDate' in all_data.columns:
        def safe_convert_date(date_str):
            try:
                return pd.to_datetime(str(date_str), format='%Y%m%d')
            except:
                return pd.NaT
        
        all_data['creatDate'] = all_data['creatDate'].apply(safe_convert_date)
        all_data['creatYear'] = all_data['creatDate'].dt.year
        all_data['creatMonth'] = all_data['creatDate'].dt.month
        all_data['creatDay'] = all_data['creatDate'].dt.day
        all_data['creatDayOfWeek'] = all_data['creatDate'].dt.dayofweek
        all_data['creatQuarter'] = all_data['creatDate'].dt.quarter
        
        # 填充缺失值
        for col in ['creatYear', 'creatMonth', 'creatDay', 'creatDayOfWeek', 'creatQuarter']:
            if col in all_data.columns:
                all_data[col].fillna(all_data[col].median(), inplace=True)
        
        # 删除原始的datetime列
        all_data.drop(['creatDate'], axis=1, inplace=True)
    
    # 5. 高级特征工程
    # 功率特征
    if 'power' in all_data.columns:
        all_data['power_squared'] = all_data['power'] ** 2
        all_data['power_sqrt'] = np.sqrt(all_data['power'])
        all_data['power_log'] = np.log1p(all_data['power'])
    
    # 公里数特征
    if 'kilometer' in all_data.columns:
        all_data['kilometer_squared'] = all_data['kilometer'] ** 2
        all_data['kilometer_sqrt'] = np.sqrt(all_data['kilometer'])
        all_data['kilometer_log'] = np.log1p(all_data['kilometer'])
    
    # 功率密度
    if 'power' in all_data.columns and 'kilometer' in all_data.columns:
        all_data['power_per_km'] = all_data['power'] / (all_data['kilometer'] + 1)
        all_data['power_km_ratio'] = all_data['power'] / (all_data['kilometer'] + 1)
    
    # 品牌-车型
    if 'brand' in all_data.columns and 'model' in all_data.columns:
        all_data['brand_model'] = all_data['brand'] * 1000 + all_data['model']
    
    # 车龄
    if 'regYear' in all_data.columns and 'creatYear' in all_data.columns:
        all_data['car_age'] = all_data['creatYear'] - all_data['regYear']
        all_data['car_age'] = all_data['car_age'].clip(0, 25)
        all_data['car_age_squared'] = all_data['car_age'] ** 2
    
    # 6. 匿名特征处理
    v_cols = [col for col in all_data.columns if col.startswith('v_')]
    if v_cols:
        all_data['v_mean'] = all_data[v_cols].mean(axis=1)
        all_data['v_std'] = all_data[v_cols].std(axis=1)
        all_data['v_max'] = all_data[v_cols].max(axis=1)
        all_data['v_min'] = all_data[v_cols].min(axis=1)
        all_data['v_median'] = all_data[v_cols].median(axis=1)
        all_data['v_sum'] = all_data[v_cols].sum(axis=1)
    
    # 7. 交互特征
    if 'power' in all_data.columns and 'car_age' in all_data.columns:
        all_data['power_age'] = all_data['power'] * all_data['car_age']
    
    if 'kilometer' in all_data.columns and 'car_age' in all_data.columns:
        all_data['km_age'] = all_data['kilometer'] * all_data['car_age']
    
    # 8. 确保所有列都是数值类型
    for col in all_data.columns:
        if all_data[col].dtype == 'object':
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
            all_data[col].fillna(all_data[col].median(), inplace=True)
    
    # 9. 特征缩放
    scaler = RobustScaler()
    numeric_features = all_data.select_dtypes(include=[np.number]).columns
    all_data[numeric_features] = scaler.fit_transform(all_data[numeric_features])
    
    print(f"特征工程完成，最终特征数量: {len(all_data.columns)}")
    print(f"特征类型: {all_data.dtypes.value_counts()}")
    
    return all_data, label_encoders, scaler

def train_multiple_models(X_train, y_train):
    """训练多个模型"""
    print("正在训练多个模型...")
    
    # 1. XGBoost - 多个版本
    xgb1 = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    xgb2 = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=123,
        n_jobs=-1
    )
    
    xgb3 = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.08,
        reg_lambda=0.8,
        random_state=456,
        n_jobs=-1
    )
    
    # 2. LightGBM
    lgb1 = lgb.LGBMRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    lgb2 = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=123,
        n_jobs=-1
    )
    
    # 3. Random Forest
    rf1 = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf2 = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=123,
        n_jobs=-1
    )
    
    # 4. Extra Trees
    et = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Gradient Boosting
    gbm = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    # 6. 线性模型
    ridge = Ridge(alpha=1.0, random_state=42)
    lasso = Lasso(alpha=0.01, random_state=42)
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    
    models = {
        'xgb1': xgb1, 'xgb2': xgb2, 'xgb3': xgb3,
        'lgb1': lgb1, 'lgb2': lgb2,
        'rf1': rf1, 'rf2': rf2,
        'et': et, 'gbm': gbm,
        'ridge': ridge, 'lasso': lasso, 'elastic': elastic
    }
    
    # 训练模型
    trained_models = {}
    cv_scores = {}
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"训练 {name}...")
        
        try:
            # 交叉验证
            scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            cv_scores[name] = {
                'mean_rmse': rmse_scores.mean(),
                'std_rmse': rmse_scores.std()
            }
            print(f"  {name} - 平均RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
            
            # 训练完整模型
            model.fit(X_train, y_train)
            trained_models[name] = model
            
        except Exception as e:
            print(f"  {name} 训练失败: {str(e)}")
            continue
    
    return trained_models, cv_scores

def weighted_ensemble_predict(models, cv_scores, X_test):
    """加权集成预测"""
    print("正在进行加权集成预测...")
    
    predictions = {}
    
    # 获取各个模型的预测
    for name, model in models.items():
        pred = model.predict(X_test)
        predictions[name] = pred
    
    # 基于交叉验证结果计算权重
    weights = {}
    total_score = sum(1 / cv_scores[name]['mean_rmse'] for name in cv_scores.keys())
    
    for name in cv_scores.keys():
        weights[name] = (1 / cv_scores[name]['mean_rmse']) / total_score
    
    print(f"模型权重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f}")
    
    # 加权集成
    ensemble_pred = np.zeros(len(X_test))
    for name, weight in weights.items():
        ensemble_pred += weight * predictions[name]
    
    return ensemble_pred, predictions, weights

def inverse_transform_predictions(predictions, transform_type):
    """逆变换预测结果"""
    print("正在进行逆变换...")
    
    if transform_type == 'log':
        # 对数变换的逆变换
        transformed_predictions = np.expm1(predictions)
    elif transform_type == 'sqrt':
        # 平方根变换的逆变换
        transformed_predictions = predictions ** 2
    else:
        transformed_predictions = predictions
    
    return transformed_predictions

def final_calibration(predictions, target_stats):
    """最终校准"""
    print("正在进行最终校准...")
    
    # 获取目标变量统计信息
    target_mean = target_stats['mean']
    target_std = target_stats['std']
    target_median = target_stats['median']
    
    # 约束预测范围
    lower_bound = max(0, target_stats['min'] - target_std)
    upper_bound = target_stats['max'] + target_std
    
    calibrated_predictions = np.clip(predictions, lower_bound, upper_bound)
    
    # 异常值处理
    extreme_threshold = 2.5 * target_std
    
    for i, pred in enumerate(calibrated_predictions):
        if abs(pred - target_mean) > extreme_threshold:
            calibrated_predictions[i] = target_median
    
    return calibrated_predictions

def main():
    """主函数"""
    print("=== 目标400分优化策略（修复版）===")
    
    # 1. 加载数据并进行目标变量变换
    train_df, test_df, original_target, transformed_target, transform_type = load_and_transform_target()
    
    # 2. 高级特征工程
    processed_data, label_encoders, scaler = advanced_feature_engineering(train_df, test_df)
    
    # 3. 分离训练集和测试集
    train_size = len(train_df)
    X_train = processed_data[:train_size]
    X_test = processed_data[train_size:]
    y_train = transformed_target
    
    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    
    # 4. 训练多个模型
    models, cv_scores = train_multiple_models(X_train, y_train)
    
    if not models:
        print("没有成功训练的模型，程序退出")
        return
    
    # 5. 加权集成预测
    ensemble_pred, individual_preds, weights = weighted_ensemble_predict(models, cv_scores, X_test)
    
    # 6. 逆变换
    transformed_predictions = inverse_transform_predictions(ensemble_pred, transform_type)
    
    # 7. 获取原始目标变量统计信息
    target_stats = {
        'mean': original_target.mean(),
        'std': original_target.std(),
        'min': original_target.min(),
        'max': original_target.max(),
        'median': original_target.median()
    }
    
    # 8. 最终校准
    final_predictions = final_calibration(transformed_predictions, target_stats)
    
    # 9. 保存结果
    submission = pd.DataFrame({
        'SaleID': range(200000, 200000 + len(final_predictions)),
        'price': final_predictions
    })
    
    submission.to_csv('target_400_optimization_fixed_submission.csv', index=False)
    print(f"预测结果已保存到 target_400_optimization_fixed_submission.csv")
    
    # 10. 输出详细统计信息
    print(f"\n=== 预测结果统计 ===")
    print(f"预测值范围: {final_predictions.min():.2f} - {final_predictions.max():.2f}")
    print(f"预测值均值: {final_predictions.mean():.2f}")
    print(f"预测值中位数: {np.median(final_predictions):.2f}")
    print(f"预测值标准差: {final_predictions.std():.2f}")
    
    # 11. 保存详细报告
    with open('target_400_optimization_fixed_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== 目标400分优化策略报告（修复版）===\n\n")
        f.write(f"目标变量变换类型: {transform_type}\n")
        
        f.write("\n模型交叉验证结果:\n")
        for name, scores in cv_scores.items():
            f.write(f"  {name}: RMSE = {scores['mean_rmse']:.4f} ± {scores['std_rmse']:.4f}\n")
        
        f.write(f"\n模型权重:\n")
        for name, weight in weights.items():
            f.write(f"  {name}: {weight:.3f}\n")
        
        f.write(f"\n原始目标变量统计:\n")
        for key, value in target_stats.items():
            f.write(f"  {key}: {value:.2f}\n")
        
        f.write(f"\n预测结果统计:\n")
        f.write(f"预测值范围: {final_predictions.min():.2f} - {final_predictions.max():.2f}\n")
        f.write(f"预测值均值: {final_predictions.mean():.2f}\n")
        f.write(f"预测值中位数: {np.median(final_predictions):.2f}\n")
        f.write(f"预测值标准差: {final_predictions.std():.2f}\n")
    
    print("目标400分优化完成！")

if __name__ == "__main__":
    main() 