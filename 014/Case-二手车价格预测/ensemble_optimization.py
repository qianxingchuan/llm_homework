import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb

# 设置随机种子
np.random.seed(42)

def load_and_preprocess_data():
    """加载数据并进行激进特征工程"""
    print("=== 激进特征工程开始 ===")
    
    # 加载数据
    train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    print(f"训练数据原始形状: {train_data.shape}")
    print(f"测试数据原始形状: {test_data.shape}")
    
    # 保存原始测试数据的SaleID
    original_test_saleids = test_data['SaleID'].copy()
    
    # 异常值过滤
    train_data = train_data[train_data['power'] < 1000]
    train_data = train_data[train_data['kilometer'] <= 20]
    train_data = train_data[(train_data['price'] > 1) & (train_data['price'] < 2e5)]
    
    # 目标变量变换
    target = train_data['price']
    print(f"\n目标变量分析:")
    print(f"  原始均值: {target.mean():.2f}")
    print(f"  原始中位数: {target.median():.2f}")
    print(f"  原始标准差: {target.std():.2f}")
    print(f"  原始偏度: {target.skew():.2f}")
    
    # 选择对数变换
    transformed_target = np.log1p(target)
    transform_type = 'log'
    print("选择对数变换")
    
    # 分离特征和目标
    train_features = train_data.drop(['price', 'SaleID'], axis=1)
    test_features = test_data.drop(['SaleID'], axis=1)
    
    # 合并数据集进行统一处理
    all_data = pd.concat([train_features, test_features], ignore_index=True)
    
    # 处理缺失值
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    # 数值型特征缺失值填充
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            if col.startswith('v_'):
                all_data[col].fillna(0, inplace=True)
            else:
                median_val = all_data[col].median()
                all_data[col].fillna(median_val, inplace=True)
    
    # 分类特征缺失值填充
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            mode_val = all_data[col].mode()[0]
            all_data[col].fillna(mode_val, inplace=True)
    
    # 时间特征处理
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
        
        for col in ['regYear', 'regMonth', 'regDay', 'regDayOfWeek', 'regQuarter']:
            if col in all_data.columns:
                all_data[col].fillna(all_data[col].median(), inplace=True)
        
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
        
        for col in ['creatYear', 'creatMonth', 'creatDay', 'creatDayOfWeek', 'creatQuarter']:
            if col in all_data.columns:
                all_data[col].fillna(all_data[col].median(), inplace=True)
        
        all_data.drop(['creatDate'], axis=1, inplace=True)
    
    # 编码分类特征
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
        label_encoders[col] = le
    
    # === 激进特征工程开始 ===
    print("开始激进特征工程...")
    
    # 1. 基础特征变换
    # 车龄特征
    if 'regYear' in all_data.columns and 'creatYear' in all_data.columns:
        all_data['car_age'] = all_data['creatYear'] - all_data['regYear']
        all_data['car_age'] = all_data['car_age'].clip(0, 25)
        all_data['car_age_squared'] = all_data['car_age'] ** 2
        all_data['car_age_cubed'] = all_data['car_age'] ** 3
        all_data['car_age_sqrt'] = np.sqrt(all_data['car_age'])
        all_data['car_age_log'] = np.log1p(all_data['car_age'])
    
    # 功率特征
    if 'power' in all_data.columns:
        all_data['power_squared'] = all_data['power'] ** 2
        all_data['power_sqrt'] = np.sqrt(all_data['power'])
        all_data['power_log'] = np.log1p(all_data['power'])
        all_data['power_inv'] = 1 / (all_data['power'] + 1)
        all_data['power_bin'] = pd.cut(all_data['power'], bins=20, labels=False).fillna(0)
        all_data['power_bin_small'] = pd.cut(all_data['power'], bins=10, labels=False).fillna(0)
    
    # 公里数特征
    if 'kilometer' in all_data.columns:
        all_data['kilometer_squared'] = all_data['kilometer'] ** 2
        all_data['kilometer_sqrt'] = np.sqrt(all_data['kilometer'])
        all_data['kilometer_log'] = np.log1p(all_data['kilometer'])
        all_data['kilometer_inv'] = 1 / (all_data['kilometer'] + 1)
        all_data['kilometer_bin'] = pd.cut(all_data['kilometer'], bins=20, labels=False).fillna(0)
        all_data['kilometer_bin_small'] = pd.cut(all_data['kilometer'], bins=10, labels=False).fillna(0)
    
    # 2. 交互特征
    if 'power' in all_data.columns and 'kilometer' in all_data.columns:
        all_data['power_per_km'] = all_data['power'] / (all_data['kilometer'] + 1)
        all_data['power_km_ratio'] = all_data['power'] / (all_data['kilometer'] + 1)
        all_data['power_km_product'] = all_data['power'] * all_data['kilometer']
        all_data['power_km_sum'] = all_data['power'] + all_data['kilometer']
        all_data['power_km_diff'] = all_data['power'] - all_data['kilometer']
    
    # 3. 品牌-车型特征
    if 'brand' in all_data.columns and 'model' in all_data.columns:
        all_data['brand_model'] = all_data['brand'] * 1000 + all_data['model']
        all_data['brand_model_ratio'] = all_data['brand'] / (all_data['model'] + 1)
        all_data['brand_model_sum'] = all_data['brand'] + all_data['model']
        all_data['brand_model_diff'] = all_data['brand'] - all_data['model']
    
    # 4. 匿名特征处理
    v_cols = [col for col in all_data.columns if col.startswith('v_')]
    if v_cols:
        all_data['v_mean'] = all_data[v_cols].mean(axis=1)
        all_data['v_std'] = all_data[v_cols].std(axis=1)
        all_data['v_max'] = all_data[v_cols].max(axis=1)
        all_data['v_min'] = all_data[v_cols].min(axis=1)
        all_data['v_median'] = all_data[v_cols].median(axis=1)
        all_data['v_sum'] = all_data[v_cols].sum(axis=1)
        all_data['v_range'] = all_data[v_cols].max(axis=1) - all_data[v_cols].min(axis=1)
        all_data['v_skew'] = all_data[v_cols].skew(axis=1).fillna(0)
        all_data['v_kurt'] = all_data[v_cols].kurt(axis=1).fillna(0)
        all_data['v_q25'] = all_data[v_cols].quantile(0.25, axis=1)
        all_data['v_q75'] = all_data[v_cols].quantile(0.75, axis=1)
        all_data['v_iqr'] = all_data['v_q75'] - all_data['v_q25']
    
    # 5. 复杂交互特征
    if 'power' in all_data.columns and 'car_age' in all_data.columns:
        all_data['power_age'] = all_data['power'] * all_data['car_age']
        all_data['power_age_ratio'] = all_data['power'] / (all_data['car_age'] + 1)
        all_data['power_age_sum'] = all_data['power'] + all_data['car_age']
        all_data['power_age_diff'] = all_data['power'] - all_data['car_age']
    
    if 'kilometer' in all_data.columns and 'car_age' in all_data.columns:
        all_data['km_age'] = all_data['kilometer'] * all_data['car_age']
        all_data['km_age_ratio'] = all_data['kilometer'] / (all_data['car_age'] + 1)
        all_data['km_age_sum'] = all_data['kilometer'] + all_data['car_age']
        all_data['km_age_diff'] = all_data['kilometer'] - all_data['car_age']
    
    if 'power' in all_data.columns and 'kilometer' in all_data.columns and 'car_age' in all_data.columns:
        all_data['power_km_age'] = all_data['power'] * all_data['kilometer'] * all_data['car_age']
        all_data['power_km_age_sum'] = all_data['power'] + all_data['kilometer'] + all_data['car_age']
    
    # 6. 品牌和车型的统计特征
    if 'brand' in all_data.columns:
        brand_stats = train_data.groupby('brand')['price'].agg(['mean', 'std', 'median', 'count']).reset_index()
        brand_stats.columns = ['brand', 'brand_price_mean', 'brand_price_std', 'brand_price_median', 'brand_count']
        all_data = all_data.merge(brand_stats, on='brand', how='left')
        
        # 填充缺失值
        for col in ['brand_price_mean', 'brand_price_std', 'brand_price_median', 'brand_count']:
            if col in all_data.columns:
                all_data[col].fillna(all_data[col].median(), inplace=True)
    
    if 'model' in all_data.columns:
        model_stats = train_data.groupby('model')['price'].agg(['mean', 'std', 'median', 'count']).reset_index()
        model_stats.columns = ['model', 'model_price_mean', 'model_price_std', 'model_price_median', 'model_count']
        all_data = all_data.merge(model_stats, on='model', how='left')
        
        # 填充缺失值
        for col in ['model_price_mean', 'model_price_std', 'model_price_median', 'model_count']:
            if col in all_data.columns:
                all_data[col].fillna(all_data[col].median(), inplace=True)
    
    # 7. 确保所有列都是数值类型
    for col in all_data.columns:
        if all_data[col].dtype == 'object':
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
            all_data[col].fillna(all_data[col].median(), inplace=True)
    
    # 8. 特征缩放
    scaler = RobustScaler()
    numeric_features = all_data.select_dtypes(include=[np.number]).columns
    all_data[numeric_features] = scaler.fit_transform(all_data[numeric_features])
    
    # 分离训练和测试数据
    train_size = len(train_data)
    X_train = all_data[:train_size]
    X_test = all_data[train_size:]
    y_train = transformed_target

    # 只保留数值型特征
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    print(f"激进特征工程完成！")
    print(f"处理后训练数据形状: {X_train.shape}")
    print(f"处理后测试数据形状: {X_test.shape}")
    print(f"最终特征数量: {len(X_train.columns)}")

    # === 特征筛选：用XGBoost筛选前60重要特征 ===
    print("\n=== 特征筛选：XGBoost重要性 ===")
    xgb_fs = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='gpu_hist',
        gpu_id=0,
        max_depth=8,
        learning_rate=0.03,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    xgb_fs.fit(X_train, y_train)
    importances = xgb_fs.feature_importances_
    feat_names = X_train.columns
    topk = 60
    top_idx = np.argsort(importances)[::-1][:topk]
    top_features = feat_names[top_idx]
    print(f"选取前{topk}个重要特征: {list(top_features)}")
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    return X_train, y_train, X_test, original_test_saleids, transform_type

def train_models(X_train, y_train):
    """训练XGBoost和LightGBM"""
    print("\n=== 开始训练XGBoost和LightGBM ===")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    models = {}
    val_scores = {}
    # XGBoost
    print("训练 XGBoost (GPU)...")
    xgb_params = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'max_depth': 8,
        'learning_rate': 0.03,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 0
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    try:
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=100)
    except TypeError:
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
    xgb_pred = xgb_model.predict(X_val)
    xgb_mae = mean_absolute_error(y_val, xgb_pred)
    print(f"XGBoost 验证集MAE: {xgb_mae:.4f}")
    models['xgb'] = xgb_model
    val_scores['xgb'] = xgb_mae
    # LightGBM
    print("训练 LightGBM (GPU)...")
    lgb_params = {
        'objective': 'regression',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'max_depth': 8,
        'learning_rate': 0.03,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbose': -1
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    lgb_pred = lgb_model.predict(X_val)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    print(f"LightGBM 验证集MAE: {lgb_mae:.4f}")
    models['lgb'] = lgb_model
    val_scores['lgb'] = lgb_mae
    return models, val_scores

def ensemble_predict(models, val_scores, X_train, y_train, X_test, transform_type):
    print("\n=== 开始集成预测 ===")
    # 只集成xgb和lgb
    weights = {}
    total_inv_score = 0
    for name in ['xgb', 'lgb']:
        score = val_scores[name]
        weights[name] = 1.0 / score
        total_inv_score += weights[name]
    for name in weights:
        weights[name] /= total_inv_score
    print("模型权重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    print("全量训练所有模型...")
    for name in ['xgb', 'lgb']:
        model = models[name]
        print(f"训练 {name}...")
        model.fit(X_train, y_train)
    predictions = {}
    for name in ['xgb', 'lgb']:
        model = models[name]
        pred = model.predict(X_test)
        predictions[name] = pred
    final_pred = np.zeros(len(X_test))
    for name, pred in predictions.items():
        final_pred += weights[name] * pred
    if transform_type == 'log':
        final_pred_original = np.expm1(final_pred)
    else:
        final_pred_original = final_pred ** 2
    return final_pred_original

def main():
    """主函数"""
    print("=== 模型融合优化版 - 冲击400分 ===")
    
    # 加载和特征工程
    X_train, y_train, X_test, test_saleids, transform_type = load_and_preprocess_data()
    
    # 训练多个模型
    models, val_scores = train_models(X_train, y_train)
    
    # 集成预测
    final_predictions = ensemble_predict(models, val_scores, X_train, y_train, X_test, transform_type)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'SaleID': test_saleids.astype(int),
        'price': final_predictions
    })
    submission.to_csv('ensemble_submission.csv', index=False)
    
    # 计算验证集集成MAE
    print("\n=== 最终结果 ===")
    print("各模型验证集MAE:")
    for name, score in val_scores.items():
        print(f"  {name}: {score:.4f}")
    
    # 估算集成后的MAE（基于权重加权平均）
    ensemble_mae = sum(score * weight for score, weight in zip(val_scores.values(), 
                                                              [1.0/score for score in val_scores.values()]))
    ensemble_mae /= sum(1.0/score for score in val_scores.values())
    
    print(f"\n预估集成后MAE: {ensemble_mae:.4f}")
    print("预测结果已保存到: ensemble_submission.csv")
    
    return models, val_scores, submission

if __name__ == "__main__":
    models, val_scores, submission = main() 