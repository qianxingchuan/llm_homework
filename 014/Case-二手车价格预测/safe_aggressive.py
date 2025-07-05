import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载和预处理数据 - 安全的激进版本"""
    print("=== 安全激进特征工程开始 ===")
    
    # 加载数据
    train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    print(f"训练数据原始形状: {train_data.shape}")
    print(f"测试数据原始形状: {test_data.shape}")
    
    # 保存原始测试数据的SaleID
    original_test_saleids = test_data['SaleID'].copy()
    
    # 基础异常值过滤
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
    
    # 对数变换
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
        
        # 填充缺失值
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
        
        # 填充缺失值
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
    
    # 安全的激进特征工程
    # 车龄特征
    if 'regYear' in all_data.columns and 'creatYear' in all_data.columns:
        all_data['car_age'] = all_data['creatYear'] - all_data['regYear']
        all_data['car_age'] = all_data['car_age'].clip(0, 25)
        all_data['car_age_squared'] = all_data['car_age'] ** 2
    
    # 功率特征
    if 'power' in all_data.columns:
        all_data['power_sqrt'] = np.sqrt(all_data['power'])
        all_data['power_log'] = np.log1p(all_data['power'])
        all_data['power_squared'] = all_data['power'] ** 2
        all_data['power_cubed'] = all_data['power'] ** 3
    
    # 公里数特征
    if 'kilometer' in all_data.columns:
        all_data['kilometer_sqrt'] = np.sqrt(all_data['kilometer'])
        all_data['kilometer_log'] = np.log1p(all_data['kilometer'])
        all_data['kilometer_squared'] = all_data['kilometer'] ** 2
    
    # 匿名特征处理
    v_cols = [col for col in all_data.columns if col.startswith('v_')]
    if v_cols:
        all_data['v_mean'] = all_data[v_cols].mean(axis=1)
        all_data['v_std'] = all_data[v_cols].std(axis=1)
        all_data['v_max'] = all_data[v_cols].max(axis=1)
        all_data['v_min'] = all_data[v_cols].min(axis=1)
        all_data['v_median'] = all_data[v_cols].median(axis=1)
        all_data['v_skew'] = all_data[v_cols].skew(axis=1)
        all_data['v_kurt'] = all_data[v_cols].kurtosis(axis=1)
        
        # 匿名特征交互
        all_data['v_mean_std_ratio'] = all_data['v_mean'] / (all_data['v_std'] + 1e-8)
        all_data['v_max_min_diff'] = all_data['v_max'] - all_data['v_min']
        all_data['v_range'] = all_data['v_max'] - all_data['v_min']
    
    # 品牌和车型特征交互
    if 'brand' in all_data.columns and 'model' in all_data.columns:
        all_data['brand_model'] = all_data['brand'].astype(str) + '_' + all_data['model'].astype(str)
        le = LabelEncoder()
        all_data['brand_model_encoded'] = le.fit_transform(all_data['brand_model'])
        all_data.drop(['brand_model'], axis=1, inplace=True)
    
    # 时间特征交互
    if 'regYear' in all_data.columns and 'regMonth' in all_data.columns:
        all_data['reg_year_month'] = all_data['regYear'] * 100 + all_data['regMonth']
    
    if 'creatYear' in all_data.columns and 'creatMonth' in all_data.columns:
        all_data['creat_year_month'] = all_data['creatYear'] * 100 + all_data['creatMonth']
    
    # 功率和公里数交互
    if 'power' in all_data.columns and 'kilometer' in all_data.columns:
        all_data['power_kilometer_ratio'] = all_data['power'] / (all_data['kilometer'] + 1e-8)
        all_data['power_kilometer_product'] = all_data['power'] * all_data['kilometer']
    
    # 车龄和公里数交互
    if 'car_age' in all_data.columns and 'kilometer' in all_data.columns:
        all_data['age_kilometer_ratio'] = all_data['car_age'] / (all_data['kilometer'] + 1e-8)
    
    # 统计特征
    if 'power' in all_data.columns:
        all_data['power_percentile_25'] = all_data['power'].quantile(0.25)
        all_data['power_percentile_75'] = all_data['power'].quantile(0.75)
        all_data['power_iqr'] = all_data['power_percentile_75'] - all_data['power_percentile_25']
    
    if 'kilometer' in all_data.columns:
        all_data['kilometer_percentile_25'] = all_data['kilometer'].quantile(0.25)
        all_data['kilometer_percentile_75'] = all_data['kilometer'].quantile(0.75)
        all_data['kilometer_iqr'] = all_data['kilometer_percentile_75'] - all_data['kilometer_percentile_25']
    
    # 确保所有列都是数值类型
    for col in all_data.columns:
        if all_data[col].dtype == 'object':
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
            all_data[col].fillna(all_data[col].median(), inplace=True)
    
    # 特征缩放
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

    print(f"安全激进特征工程完成！")
    print(f"处理后训练数据形状: {X_train.shape}")
    print(f"处理后测试数据形状: {X_test.shape}")
    print(f"最终特征数量: {len(X_train.columns)}")
    
    return X_train, y_train, X_test, original_test_saleids, transform_type

def train_and_predict(X_train, y_train, X_test, test_saleids, transform_type):
    """训练和预测 - 安全的激进参数"""
    print("\n=== 开始安全激进XGBoost训练 ===")
    
    # 确保所有特征为float类型
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    # 划分验证集
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    # 安全的激进XGBoost参数
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 2000,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 0
    }
    
    print("训练 XGBoost (安全激进参数)...")
    model = xgb.XGBRegressor(**params)
    try:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=100)
    except TypeError:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
    
    val_pred = model.predict(X_val)
    
    # 逆变换验证集预测结果
    if transform_type == 'log':
        val_pred_original = np.expm1(val_pred)
        y_val_original = np.expm1(y_val)
    else:  # sqrt
        val_pred_original = val_pred ** 2
        y_val_original = y_val ** 2
    
    val_mae = mean_absolute_error(y_val_original, val_pred_original)
    print(f"XGBoost 验证集MAE: {val_mae:.2f}")
    
    # 全量训练
    print("全量训练XGBoost...")
    model.fit(X_train, y_train, verbose=False)
    test_pred = model.predict(X_test)
    
    # 逆变换测试集预测结果
    if transform_type == 'log':
        test_pred_original = np.expm1(test_pred)
    else:  # sqrt
        test_pred_original = test_pred ** 2
    
    # 生成提交文件
    submission = pd.DataFrame({
        'SaleID': test_saleids.astype(int),
        'price': test_pred_original
    })
    submission.to_csv('safe_aggressive_submission.csv', index=False)
    
    print(f"\n=== 最终结果 ===")
    print(f"XGBoost 验证集MAE: {val_mae:.2f}")
    print("预测结果已保存到: safe_aggressive_submission.csv")
    
    return val_mae, submission

if __name__ == "__main__":
    print("=== 安全激进XGBoost ===")
    X_train, y_train, X_test, test_saleids, transform_type = load_and_preprocess_data()
    val_mae, submission = train_and_predict(X_train, y_train, X_test, test_saleids, transform_type)
    print(f"最终验证集MAE: {val_mae:.2f}") 