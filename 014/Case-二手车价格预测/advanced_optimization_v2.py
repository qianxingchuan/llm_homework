import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# 设置随机种子
np.random.seed(42)

def load_and_analyze_data():
    """加载和分析数据"""
    print("正在加载和分析数据...")
    
    train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    # 分析目标变量分布
    target = train_df['price']
    print(f"\n目标变量统计:")
    print(f"  均值: {target.mean():.2f}")
    print(f"  中位数: {target.median():.2f}")
    print(f"  标准差: {target.std():.2f}")
    print(f"  最小值: {target.min():.2f}")
    print(f"  最大值: {target.max():.2f}")
    print(f"  25%分位数: {target.quantile(0.25):.2f}")
    print(f"  75%分位数: {target.quantile(0.75):.2f}")
    
    return train_df, test_df, target

def advanced_preprocessing(train_df, test_df):
    """高级预处理"""
    print("正在进行高级预处理...")
    
    # 分离特征和目标
    target = train_df['price']
    train_features = train_df.drop(['price', 'SaleID'], axis=1)
    test_features = test_df.drop(['SaleID'], axis=1)
    
    # 合并数据集
    all_data = pd.concat([train_features, test_features], ignore_index=True)
    
    # 1. 处理缺失值
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    # 数值型特征：使用更智能的填充方法
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            if col.startswith('v_'):  # 匿名特征
                # 匿名特征用0填充
                all_data[col].fillna(0, inplace=True)
            else:
                # 其他特征用中位数填充
                median_val = all_data[col].median()
                all_data[col].fillna(median_val, inplace=True)
    
    # 分类特征：用众数填充
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            mode_val = all_data[col].mode()[0]
            all_data[col].fillna(mode_val, inplace=True)
    
    # 2. 异常值处理 - 使用更保守的方法
    for col in numeric_cols:
        if col.startswith('v_'):  # 跳过匿名特征
            continue
        
        # 使用更宽松的分位数
        Q1 = all_data[col].quantile(0.01)  # 1%分位数
        Q3 = all_data[col].quantile(0.99)  # 99%分位数
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 更宽松的边界
        upper_bound = Q3 + 3 * IQR
        
        all_data[col] = all_data[col].clip(lower_bound, upper_bound)
    
    # 3. 编码分类特征
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
        label_encoders[col] = le
    
    # 4. 高级特征工程
    # 功率密度
    if 'power' in all_data.columns and 'kilometer' in all_data.columns:
        all_data['power_density'] = all_data['power'] / (all_data['kilometer'] + 1)
    
    # 品牌-车型组合
    if 'brand' in all_data.columns and 'model' in all_data.columns:
        all_data['brand_model'] = all_data['brand'] * 1000 + all_data['model']
    
    # 时间特征
    if 'regDate' in all_data.columns:
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
    
    # 车龄
    if 'regYear' in all_data.columns and 'creatYear' in all_data.columns:
        all_data['car_age'] = all_data['creatYear'] - all_data['regYear']
        all_data['car_age'] = all_data['car_age'].clip(0, 30)
    
    # 5. 特征缩放 - 使用RobustScaler对异常值更稳健
    scaler = RobustScaler()
    numeric_features = all_data.select_dtypes(include=[np.number]).columns
    all_data[numeric_features] = scaler.fit_transform(all_data[numeric_features])
    
    return all_data, target, label_encoders, scaler

def train_ensemble_models(X_train, y_train):
    """训练集成模型"""
    print("正在训练集成模型...")
    
    # 1. 基础模型
    models = {
        'xgb1': xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        ),
        'xgb2': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=123,
            n_jobs=-1
        ),
        'lgb': lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        ),
        'rf': RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'gbm': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'ridge': Ridge(alpha=10.0, random_state=42)
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
        print(f"  {name} - 平均RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
        
        # 训练完整模型
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # 3. 创建加权集成
    # 基于交叉验证结果分配权重
    weights = {}
    total_score = sum(1 / cv_scores[name]['mean_rmse'] for name in cv_scores.keys())
    
    for name in cv_scores.keys():
        weights[name] = (1 / cv_scores[name]['mean_rmse']) / total_score
    
    print(f"\n模型权重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f}")
    
    return trained_models, cv_scores, weights

def ensemble_predict(models, weights, X_test):
    """集成预测"""
    print("正在进行集成预测...")
    
    predictions = {}
    
    # 获取各个模型的预测
    for name, model in models.items():
        pred = model.predict(X_test)
        predictions[name] = pred
    
    # 加权集成
    ensemble_pred = np.zeros(len(X_test))
    for name, weight in weights.items():
        ensemble_pred += weight * predictions[name]
    
    return ensemble_pred, predictions

def calibrate_predictions(predictions, target_stats, X_train, y_train):
    """预测校准"""
    print("正在进行预测校准...")
    
    # 1. 基于训练集分布进行校准
    target_mean = target_stats['mean']
    target_std = target_stats['std']
    target_median = target_stats['median']
    
    # 2. 计算预测值的统计信息
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    
    # 3. 标准化预测值到目标分布
    # 使用Z-score标准化，然后重新映射到目标分布
    z_scores = (predictions - pred_mean) / pred_std
    calibrated_predictions = z_scores * target_std + target_mean
    
    # 4. 应用约束
    # 确保预测值在合理范围内
    lower_bound = max(0, target_stats['min'] - 2 * target_std)
    upper_bound = target_stats['max'] + 2 * target_std
    
    calibrated_predictions = np.clip(calibrated_predictions, lower_bound, upper_bound)
    
    # 5. 异常值处理
    # 如果预测值过于极端，使用中位数
    extreme_threshold = 2.5 * target_std
    
    for i, pred in enumerate(calibrated_predictions):
        if abs(pred - target_mean) > extreme_threshold:
            # 使用中位数替代极端值
            calibrated_predictions[i] = target_median
    
    return calibrated_predictions

def main():
    """主函数"""
    print("=== 高级优化策略 V2 ===")
    
    # 1. 加载和分析数据
    train_df, test_df, target = load_and_analyze_data()
    
    # 2. 高级预处理
    processed_data, target, label_encoders, scaler = advanced_preprocessing(train_df, test_df)
    
    # 3. 分离训练集和测试集
    train_size = len(train_df)
    X_train = processed_data[:train_size]
    X_test = processed_data[train_size:]
    
    # 4. 获取目标变量统计信息
    target_stats = {
        'mean': target.mean(),
        'std': target.std(),
        'min': target.min(),
        'max': target.max(),
        'median': target.median(),
        'q25': target.quantile(0.25),
        'q75': target.quantile(0.75)
    }
    
    # 5. 训练集成模型
    models, cv_scores, weights = train_ensemble_models(X_train, target)
    
    # 6. 集成预测
    ensemble_pred, individual_preds = ensemble_predict(models, weights, X_test)
    
    # 7. 预测校准
    final_predictions = calibrate_predictions(ensemble_pred, target_stats, X_train, target)
    
    # 8. 保存结果
    submission = pd.DataFrame({
        'SaleID': range(200000, 200000 + len(final_predictions)),
        'price': final_predictions
    })
    
    submission.to_csv('advanced_optimization_v2_submission.csv', index=False)
    print(f"预测结果已保存到 advanced_optimization_v2_submission.csv")
    
    # 9. 输出详细统计信息
    print(f"\n=== 预测结果统计 ===")
    print(f"预测值范围: {final_predictions.min():.2f} - {final_predictions.max():.2f}")
    print(f"预测值均值: {final_predictions.mean():.2f}")
    print(f"预测值中位数: {np.median(final_predictions):.2f}")
    print(f"预测值标准差: {final_predictions.std():.2f}")
    print(f"预测值25%分位数: {np.percentile(final_predictions, 25):.2f}")
    print(f"预测值75%分位数: {np.percentile(final_predictions, 75):.2f}")
    
    # 10. 保存详细报告
    with open('advanced_optimization_v2_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== 高级优化策略 V2 报告 ===\n\n")
        f.write("模型交叉验证结果:\n")
        for name, scores in cv_scores.items():
            f.write(f"  {name}: RMSE = {scores['mean_rmse']:.2f} ± {scores['std_rmse']:.2f}\n")
        
        f.write(f"\n模型权重:\n")
        for name, weight in weights.items():
            f.write(f"  {name}: {weight:.3f}\n")
        
        f.write(f"\n目标变量统计:\n")
        for key, value in target_stats.items():
            f.write(f"  {key}: {value:.2f}\n")
        
        f.write(f"\n预测结果统计:\n")
        f.write(f"预测值范围: {final_predictions.min():.2f} - {final_predictions.max():.2f}\n")
        f.write(f"预测值均值: {final_predictions.mean():.2f}\n")
        f.write(f"预测值中位数: {np.median(final_predictions):.2f}\n")
        f.write(f"预测值标准差: {final_predictions.std():.2f}\n")
        f.write(f"预测值25%分位数: {np.percentile(final_predictions, 25):.2f}\n")
        f.write(f"预测值75%分位数: {np.percentile(final_predictions, 75):.2f}\n")
    
    print("高级优化完成！")

if __name__ == "__main__":
    main() 