import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("加载数据...")
    
    # 使用更稳健的数据读取方式
    train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    print(f"训练数据原始形状: {train_data.shape}")
    print(f"测试数据原始形状: {test_data.shape}")
    
    # 保存原始测试数据的SaleID
    original_test_saleids = test_data['SaleID'].copy()
    
    # 更保守的异常值过滤
    train_data = train_data[train_data['power'] < 1000]
    train_data = train_data[train_data['kilometer'] <= 20]
    train_data = train_data[(train_data['price'] > 1) & (train_data['price'] < 2e5)]
    
    # 目标变量变换 - 关键优化！（移动到异常值过滤后）
    target = train_data['price']
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
    
    # 高级特征工程 - 参考target_400的成功经验
    # 车龄特征
    if 'regYear' in all_data.columns and 'creatYear' in all_data.columns:
        all_data['car_age'] = all_data['creatYear'] - all_data['regYear']
        all_data['car_age'] = all_data['car_age'].clip(0, 25)
        all_data['car_age_squared'] = all_data['car_age'] ** 2
        all_data['car_age_cubed'] = all_data['car_age'] ** 3
    
    # 功率特征 - 更丰富的变换
    if 'power' in all_data.columns:
        all_data['power_squared'] = all_data['power'] ** 2
        all_data['power_sqrt'] = np.sqrt(all_data['power'])
        all_data['power_log'] = np.log1p(all_data['power'])
        all_data['power_inv'] = 1 / (all_data['power'] + 1)
        all_data['power_bin'] = pd.cut(all_data['power'], bins=10, labels=False).fillna(0)
    
    # 公里数特征 - 更丰富的变换
    if 'kilometer' in all_data.columns:
        all_data['kilometer_squared'] = all_data['kilometer'] ** 2
        all_data['kilometer_sqrt'] = np.sqrt(all_data['kilometer'])
        all_data['kilometer_log'] = np.log1p(all_data['kilometer'])
        all_data['kilometer_inv'] = 1 / (all_data['kilometer'] + 1)
        all_data['kilometer_bin'] = pd.cut(all_data['kilometer'], bins=10, labels=False).fillna(0)
    
    # 功率密度和比率特征
    if 'power' in all_data.columns and 'kilometer' in all_data.columns:
        all_data['power_per_km'] = all_data['power'] / (all_data['kilometer'] + 1)
        all_data['power_km_ratio'] = all_data['power'] / (all_data['kilometer'] + 1)
        all_data['power_km_product'] = all_data['power'] * all_data['kilometer']
    
    # 品牌-车型组合特征
    if 'brand' in all_data.columns and 'model' in all_data.columns:
        all_data['brand_model'] = all_data['brand'] * 1000 + all_data['model']
        all_data['brand_model_ratio'] = all_data['brand'] / (all_data['model'] + 1)
    
    # 匿名特征处理 - 更丰富的统计特征
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
    
    # 交互特征 - 更多组合
    if 'power' in all_data.columns and 'car_age' in all_data.columns:
        all_data['power_age'] = all_data['power'] * all_data['car_age']
        all_data['power_age_ratio'] = all_data['power'] / (all_data['car_age'] + 1)
    
    if 'kilometer' in all_data.columns and 'car_age' in all_data.columns:
        all_data['km_age'] = all_data['kilometer'] * all_data['car_age']
        all_data['km_age_ratio'] = all_data['kilometer'] / (all_data['car_age'] + 1)
    
    if 'power' in all_data.columns and 'kilometer' in all_data.columns and 'car_age' in all_data.columns:
        all_data['power_km_age'] = all_data['power'] * all_data['kilometer'] * all_data['car_age']
    
    # 确保所有列都是数值类型
    for col in all_data.columns:
        if all_data[col].dtype == 'object':
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
            all_data[col].fillna(all_data[col].median(), inplace=True)
    
    # 特征缩放 - 关键优化！
    scaler = RobustScaler()
    numeric_features = all_data.select_dtypes(include=[np.number]).columns
    all_data[numeric_features] = scaler.fit_transform(all_data[numeric_features])
    
    # 分离训练和测试数据
    train_size = len(train_data)
    X_train = all_data[:train_size]
    X_test = all_data[train_size:]
    y_train = transformed_target  # 使用变换后的目标变量

    # 只保留数值型特征
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    print(f"处理后训练数据形状: {X_train.shape}")
    print(f"处理后测试数据形状: {X_test.shape}")
    print(f"特征工程完成，最终特征数量: {len(X_train.columns)}")
    
    return X_train, y_train, X_test, original_test_saleids, transform_type

def train_and_predict(X_train, y_train, X_test, test_saleids, transform_type):
    print("开始XGBoost(GPU)训练...")
    # 确保所有特征为float类型
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    # 划分验证集
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    # 优化的XGBoost参数 
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'max_depth': 8,  # 增加深度
        'learning_rate': 0.05,  # 降低学习率
        'n_estimators': 3000,  # 增加树的数量
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,  # 添加最小子节点权重
        'reg_alpha': 0.1,  # L1正则化
        'reg_lambda': 1.0,  # L2正则化
        'random_state': 42,
        'verbosity': 0
    }
    model = xgb.XGBRegressor(**params)
    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            early_stopping_rounds=100,  # 增加早停轮数
            verbose=100
        )
    except TypeError as e:
        print(f"fit参数不兼容，降级为无早停训练: {e}")
        model.fit(X_tr, y_tr, verbose=100)
    val_pred = model.predict(X_val)
    
    # 逆变换验证集预测结果
    if transform_type == 'log':
        val_pred_original = np.expm1(val_pred)
        y_val_original = np.expm1(y_val)
    else:  # sqrt
        val_pred_original = val_pred ** 2
        y_val_original = y_val ** 2
    
    val_mae = mean_absolute_error(y_val_original, val_pred_original)
    print(f"验证集MAE: {val_mae:.4f}")
    
    # 全量训练
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
    submission.to_csv('gpu_accelerated_submission_fixed.csv', index=False)
    print("预测结果已保存到: gpu_accelerated_submission_fixed.csv")
    return val_mae, submission

if __name__ == "__main__":
    print("=== GPU加速二手车价格预测 (优化版) ===")
    X_train, y_train, X_test, test_saleids, transform_type = load_and_preprocess_data()
    print(f"训练数据: {X_train.shape}, 测试数据: {X_test.shape}")
    val_mae, submission = train_and_predict(X_train, y_train, X_test, test_saleids, transform_type)
    print(f"最终验证集MAE: {val_mae:.4f}") 