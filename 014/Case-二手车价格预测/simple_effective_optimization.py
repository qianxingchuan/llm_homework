import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# 设置随机种子
np.random.seed(42)

def load_data():
    """加载数据"""
    print("正在加载数据...")
    train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    return train_df, test_df

def simple_preprocessing(train_df, test_df):
    """简单但有效的预处理"""
    print("正在进行数据预处理...")
    
    # 分离特征和目标
    target = train_df['price']
    train_features = train_df.drop(['price', 'SaleID'], axis=1)
    test_features = test_df.drop(['SaleID'], axis=1)
    
    # 合并数据集进行一致处理
    all_data = pd.concat([train_features, test_features], ignore_index=True)
    
    # 1. 处理缺失值
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    
    # 数值型特征用中位数填充
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            median_val = all_data[col].median()
            all_data[col].fillna(median_val, inplace=True)
    
    # 分类特征用众数填充
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            mode_val = all_data[col].mode()[0]
            all_data[col].fillna(mode_val, inplace=True)
    
    # 2. 处理异常值 - 使用更保守的方法
    for col in numeric_cols:
        if col.startswith('v_'):  # 匿名特征
            continue  # 跳过匿名特征的处理
        
        Q1 = all_data[col].quantile(0.05)  # 使用5%和95%分位数
        Q3 = all_data[col].quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR  # 更宽松的边界
        upper_bound = Q3 + 2 * IQR
        
        all_data[col] = all_data[col].clip(lower_bound, upper_bound)
    
    # 3. 编码分类特征
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
        label_encoders[col] = le
    
    # 4. 添加简单的特征
    if 'power' in all_data.columns and 'kilometer' in all_data.columns:
        all_data['power_per_km'] = all_data['power'] / (all_data['kilometer'] + 1)
    
    # 5. 时间特征处理
    if 'regDate' in all_data.columns:
        def safe_convert_date(date_str):
            try:
                return pd.to_datetime(str(date_str), format='%Y%m%d')
            except:
                return pd.NaT
        
        all_data['regDate'] = all_data['regDate'].apply(safe_convert_date)
        all_data['regYear'] = all_data['regDate'].dt.year
        all_data['regYear'].fillna(all_data['regYear'].median(), inplace=True)
    
    if 'creatDate' in all_data.columns:
        all_data['creatDate'] = all_data['creatDate'].apply(safe_convert_date)
        all_data['creatYear'] = all_data['creatDate'].dt.year
        all_data['creatYear'].fillna(all_data['creatYear'].median(), inplace=True)
    
    # 计算车龄
    if 'regYear' in all_data.columns and 'creatYear' in all_data.columns:
        all_data['car_age'] = all_data['creatYear'] - all_data['regYear']
        all_data['car_age'] = all_data['car_age'].clip(0, 25)  # 限制车龄
    
    return all_data, target

def train_stable_model(X_train, y_train):
    """训练稳定的模型"""
    print("正在训练模型...")
    
    # 使用XGBoost，但参数更保守
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,  # 降低深度防止过拟合
        learning_rate=0.05,  # 降低学习率
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1正则化
        reg_lambda=1.0,  # L2正则化
        random_state=42,
        n_jobs=-1
    )
    
    # 交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    
    print(f"交叉验证RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
    
    # 训练完整模型
    model.fit(X_train, y_train)
    
    return model, rmse_scores.mean()

def predict_with_constraints(model, X_test, target_stats):
    """带约束的预测"""
    print("正在进行预测...")
    
    # 基础预测
    predictions = model.predict(X_test)
    
    # 获取目标变量统计信息
    target_mean = target_stats['mean']
    target_std = target_stats['std']
    target_min = target_stats['min']
    target_max = target_stats['max']
    
    # 约束预测范围
    # 使用训练集的统计信息，但允许一定的灵活性
    lower_bound = max(0, target_min - 1.5 * target_std)
    upper_bound = target_max + 1.5 * target_std
    
    # 应用约束
    constrained_predictions = np.clip(predictions, lower_bound, upper_bound)
    
    # 额外的合理性检查
    # 如果预测值过于极端，使用中位数
    median_price = target_stats['median']
    extreme_threshold = 3 * target_std
    
    for i, pred in enumerate(constrained_predictions):
        if abs(pred - target_mean) > extreme_threshold:
            # 如果预测值偏离均值太远，使用中位数
            constrained_predictions[i] = median_price
    
    return constrained_predictions

def main():
    """主函数"""
    print("=== 简单有效优化策略 ===")
    
    # 1. 加载数据
    train_df, test_df = load_data()
    
    # 2. 预处理
    processed_data, target = simple_preprocessing(train_df, test_df)
    
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
        'median': target.median()
    }
    
    print(f"目标变量统计:")
    for key, value in target_stats.items():
        print(f"  {key}: {value:.2f}")
    
    # 5. 训练模型
    model, cv_rmse = train_stable_model(X_train, target)
    
    # 6. 预测
    predictions = predict_with_constraints(model, X_test, target_stats)
    
    # 7. 保存结果
    submission = pd.DataFrame({
        'SaleID': range(200000, 200000 + len(predictions)),
        'price': predictions
    })
    
    submission.to_csv('simple_effective_submission.csv', index=False)
    print(f"预测结果已保存到 simple_effective_submission.csv")
    
    # 8. 输出预测统计
    print(f"\n预测结果统计:")
    print(f"预测值范围: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"预测值均值: {predictions.mean():.2f}")
    print(f"预测值中位数: {np.median(predictions):.2f}")
    print(f"预测值标准差: {predictions.std():.2f}")
    
    # 9. 保存报告
    with open('simple_effective_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== 简单有效优化策略报告 ===\n\n")
        f.write(f"交叉验证RMSE: {cv_rmse:.2f}\n")
        f.write(f"目标变量统计:\n")
        for key, value in target_stats.items():
            f.write(f"  {key}: {value:.2f}\n")
        f.write(f"\n预测结果统计:\n")
        f.write(f"预测值范围: {predictions.min():.2f} - {predictions.max():.2f}\n")
        f.write(f"预测值均值: {predictions.mean():.2f}\n")
        f.write(f"预测值中位数: {np.median(predictions):.2f}\n")
        f.write(f"预测值标准差: {predictions.std():.2f}\n")
    
    print("优化完成！")

if __name__ == "__main__":
    main() 