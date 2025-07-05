import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import optuna

# 设置随机种子
np.random.seed(42)

def load_and_analyze_data():
    """加载和深度分析数据"""
    print("正在加载和深度分析数据...")
    
    train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    # 深度分析目标变量
    target = train_df['price']
    print(f"\n目标变量深度分析:")
    print(f"  均值: {target.mean():.2f}")
    print(f"  中位数: {target.median():.2f}")
    print(f"  标准差: {target.std():.2f}")
    print(f"  偏度: {target.skew():.2f}")
    print(f"  峰度: {target.kurtosis():.2f}")
    
    # 检查目标变量分布
    print(f"  对数变换后偏度: {np.log1p(target).skew():.2f}")
    
    return train_df, test_df, target

def advanced_feature_engineering(train_df, test_df):
    """高级特征工程"""
    print("正在进行高级特征工程...")
    
    # 分离特征和目标
    target = train_df['price']
    train_features = train_df.drop(['price', 'SaleID'], axis=1)
    test_features = test_df.drop(['SaleID'], axis=1)
    
    # 合并数据集
    all_data = pd.concat([train_features, test_features], ignore_index=True)
    
    # 1. 智能缺失值处理
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    # 数值型特征：使用更智能的填充
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            if col.startswith('v_'):
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
    
    # 2. 高级异常值处理
    for col in numeric_cols:
        if col.startswith('v_'):
            continue
        
        # 使用IQR方法，但更智能
        Q1 = all_data[col].quantile(0.25)
        Q3 = all_data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 计算异常值边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 只处理极端异常值，保留一些变异
        extreme_lower = Q1 - 3 * IQR
        extreme_upper = Q3 + 3 * IQR
        
        # 将极端异常值限制在合理范围内
        all_data[col] = all_data[col].clip(extreme_lower, extreme_upper)
    
    # 3. 编码分类特征
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
        label_encoders[col] = le
    
    # 4. 高级特征工程
    # 功率相关特征
    if 'power' in all_data.columns:
        all_data['power_squared'] = all_data['power'] ** 2
        all_data['power_sqrt'] = np.sqrt(all_data['power'])
        all_data['power_log'] = np.log1p(all_data['power'])
    
    # 公里数相关特征
    if 'kilometer' in all_data.columns:
        all_data['kilometer_squared'] = all_data['kilometer'] ** 2
        all_data['kilometer_sqrt'] = np.sqrt(all_data['kilometer'])
        all_data['kilometer_log'] = np.log1p(all_data['kilometer'])
    
    # 功率密度特征
    if 'power' in all_data.columns and 'kilometer' in all_data.columns:
        all_data['power_per_km'] = all_data['power'] / (all_data['kilometer'] + 1)
        all_data['power_km_ratio'] = all_data['power'] / (all_data['kilometer'] + 1)
        all_data['power_km_product'] = all_data['power'] * all_data['kilometer']
    
    # 品牌-车型组合特征
    if 'brand' in all_data.columns and 'model' in all_data.columns:
        all_data['brand_model'] = all_data['brand'] * 1000 + all_data['model']
        all_data['brand_model_ratio'] = all_data['brand'] / (all_data['model'] + 1)
    
    # 5. 时间特征工程
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
    
    if 'creatDate' in all_data.columns:
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
    
    # 车龄相关特征
    if 'regYear' in all_data.columns and 'creatYear' in all_data.columns:
        all_data['car_age'] = all_data['creatYear'] - all_data['regYear']
        all_data['car_age'] = all_data['car_age'].clip(0, 30)
        all_data['car_age_squared'] = all_data['car_age'] ** 2
        all_data['car_age_sqrt'] = np.sqrt(all_data['car_age'])
    
    # 6. 匿名特征处理
    v_cols = [col for col in all_data.columns if col.startswith('v_')]
    if v_cols:
        # 匿名特征的统计特征
        all_data['v_mean'] = all_data[v_cols].mean(axis=1)
        all_data['v_std'] = all_data[v_cols].std(axis=1)
        all_data['v_max'] = all_data[v_cols].max(axis=1)
        all_data['v_min'] = all_data[v_cols].min(axis=1)
        all_data['v_median'] = all_data[v_cols].median(axis=1)
        all_data['v_sum'] = all_data[v_cols].sum(axis=1)
        
        # 匿名特征的主成分
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3, random_state=42)
        v_pca = pca.fit_transform(all_data[v_cols])
        all_data['v_pca1'] = v_pca[:, 0]
        all_data['v_pca2'] = v_pca[:, 1]
        all_data['v_pca3'] = v_pca[:, 2]
    
    # 7. 交互特征
    if 'power' in all_data.columns and 'car_age' in all_data.columns:
        all_data['power_age'] = all_data['power'] * all_data['car_age']
    
    if 'kilometer' in all_data.columns and 'car_age' in all_data.columns:
        all_data['km_age'] = all_data['kilometer'] * all_data['car_age']
    
    # 8. 特征缩放 - 使用PowerTransformer处理偏度
    numeric_features = all_data.select_dtypes(include=[np.number]).columns
    
    # 对偏度较大的特征使用PowerTransformer
    skewed_features = []
    for col in numeric_features:
        if abs(all_data[col].skew()) > 1:
            skewed_features.append(col)
    
    if skewed_features:
        pt = PowerTransformer(method='yeo-johnson')
        all_data[skewed_features] = pt.fit_transform(all_data[skewed_features])
    
    # 对其他特征使用StandardScaler
    other_features = [col for col in numeric_features if col not in skewed_features]
    if other_features:
        scaler = StandardScaler()
        all_data[other_features] = scaler.fit_transform(all_data[other_features])
    
    return all_data, target, label_encoders

def feature_selection(X_train, y_train, X_test):
    """特征选择"""
    print("正在进行特征选择...")
    
    # 1. 基于F统计量的特征选择
    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X_train, y_train)
    
    # 获取特征重要性分数
    feature_scores = pd.DataFrame({
        'feature': X_train.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    print("Top 20 特征重要性:")
    print(feature_scores.head(20))
    
    # 选择前80%的特征
    n_features = int(len(X_train.columns) * 0.8)
    selected_features = feature_scores.head(n_features)['feature'].tolist()
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    print(f"选择了 {len(selected_features)} 个特征")
    
    return X_train_selected, X_test_selected, selected_features

def optimize_hyperparameters(X_train, y_train):
    """超参数优化"""
    print("正在进行超参数优化...")
    
    def objective(trial):
        # XGBoost参数
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
            'random_state': 42
        }
        
        # 交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            xgb.XGBRegressor(**xgb_params), 
            X_train, y_train, 
            cv=kfold, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return np.sqrt(-scores.mean())
    
    # 运行优化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print(f"最佳参数: {study.best_params}")
    print(f"最佳RMSE: {study.best_value:.2f}")
    
    return study.best_params

def train_optimized_models(X_train, y_train, best_params):
    """训练优化后的模型"""
    print("正在训练优化后的模型...")
    
    # 1. 优化的XGBoost
    xgb_model = xgb.XGBRegressor(**best_params, n_jobs=-1)
    
    # 2. LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 4. Extra Trees
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Gradient Boosting
    gbm_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    # 6. Ridge回归
    ridge_model = Ridge(alpha=1.0, random_state=42)
    
    # 7. Huber回归（对异常值更稳健）
    huber_model = HuberRegressor(epsilon=1.35, max_iter=1000)
    
    models = {
        'xgb': xgb_model,
        'lgb': lgb_model,
        'rf': rf_model,
        'et': et_model,
        'gbm': gbm_model,
        'ridge': ridge_model,
        'huber': huber_model
    }
    
    # 训练模型
    trained_models = {}
    cv_scores = {}
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"训练 {name}...")
        
        # 交叉验证
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
    
    return trained_models, cv_scores

def ensemble_predict_with_optimization(models, cv_scores, X_test):
    """优化的集成预测"""
    print("正在进行优化的集成预测...")
    
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

def advanced_post_processing(predictions, target_stats, X_train, y_train):
    """高级后处理"""
    print("正在进行高级后处理...")
    
    # 1. 基于训练集分布进行校准
    target_mean = target_stats['mean']
    target_std = target_stats['std']
    target_median = target_stats['median']
    
    # 2. 计算预测值的统计信息
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    
    # 3. 标准化预测值到目标分布
    z_scores = (predictions - pred_mean) / pred_std
    calibrated_predictions = z_scores * target_std + target_mean
    
    # 4. 应用更严格的约束
    # 使用训练集的1%和99%分位数作为边界
    lower_bound = max(0, target_stats['min'])
    upper_bound = target_stats['max'] + target_std
    
    calibrated_predictions = np.clip(calibrated_predictions, lower_bound, upper_bound)
    
    # 5. 异常值处理
    # 使用更严格的阈值
    extreme_threshold = 2 * target_std
    
    for i, pred in enumerate(calibrated_predictions):
        if abs(pred - target_mean) > extreme_threshold:
            # 使用中位数替代极端值
            calibrated_predictions[i] = target_median
    
    # 6. 平滑处理
    # 使用移动中位数平滑
    window_size = 3
    if len(calibrated_predictions) > window_size:
        smoothed = pd.Series(calibrated_predictions).rolling(window=window_size, center=True).median().fillna(method='bfill').fillna(method='ffill')
        # 只对中间部分使用平滑
        calibrated_predictions[window_size:-window_size] = smoothed[window_size:-window_size]
    
    return calibrated_predictions

def main():
    """主函数"""
    print("=== 高分优化策略 ===")
    
    # 1. 加载和深度分析数据
    train_df, test_df, target = load_and_analyze_data()
    
    # 2. 高级特征工程
    processed_data, target, label_encoders = advanced_feature_engineering(train_df, test_df)
    
    # 3. 分离训练集和测试集
    train_size = len(train_df)
    X_train = processed_data[:train_size]
    X_test = processed_data[train_size:]
    
    # 4. 特征选择
    X_train_selected, X_test_selected, selected_features = feature_selection(X_train, target, X_test)
    
    # 5. 超参数优化
    best_params = optimize_hyperparameters(X_train_selected, target)
    
    # 6. 训练优化后的模型
    models, cv_scores = train_optimized_models(X_train_selected, target, best_params)
    
    # 7. 优化的集成预测
    ensemble_pred, individual_preds, weights = ensemble_predict_with_optimization(models, cv_scores, X_test_selected)
    
    # 8. 获取目标变量统计信息
    target_stats = {
        'mean': target.mean(),
        'std': target.std(),
        'min': target.min(),
        'max': target.max(),
        'median': target.median(),
        'q25': target.quantile(0.25),
        'q75': target.quantile(0.75)
    }
    
    # 9. 高级后处理
    final_predictions = advanced_post_processing(ensemble_pred, target_stats, X_train_selected, target)
    
    # 10. 保存结果
    submission = pd.DataFrame({
        'SaleID': range(200000, 200000 + len(final_predictions)),
        'price': final_predictions
    })
    
    submission.to_csv('high_score_optimization_submission.csv', index=False)
    print(f"预测结果已保存到 high_score_optimization_submission.csv")
    
    # 11. 输出详细统计信息
    print(f"\n=== 预测结果统计 ===")
    print(f"预测值范围: {final_predictions.min():.2f} - {final_predictions.max():.2f}")
    print(f"预测值均值: {final_predictions.mean():.2f}")
    print(f"预测值中位数: {np.median(final_predictions):.2f}")
    print(f"预测值标准差: {final_predictions.std():.2f}")
    
    # 12. 保存详细报告
    with open('high_score_optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== 高分优化策略报告 ===\n\n")
        f.write("超参数优化结果:\n")
        f.write(f"最佳参数: {best_params}\n")
        
        f.write("\n模型交叉验证结果:\n")
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
        
        f.write(f"\n选择的特征数量: {len(selected_features)}\n")
        f.write(f"特征列表: {selected_features}\n")
    
    print("高分优化完成！")

if __name__ == "__main__":
    main() 