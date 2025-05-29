import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取数据
df = pd.read_excel(os.path.join(script_dir, 'policy_data.xlsx'))

# 数据预处理
def preprocess_data(df):
    # 复制数据框
    df_processed = df.copy()
    
    # 对类别变量进行编码
    categorical_cols = ['gender', 'birth_region', 'insurance_region', 'income_level',
                       'education_level', 'occupation', 'marital_status',
                       'policy_type', 'claim_history']
    
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df_processed[col] = label_encoders[col].fit_transform(df_processed[col])
    
    # 将目标变量转换为数值
    df_processed['renewal'] = (df_processed['renewal'] == 'Yes').astype(int)
    
    return df_processed, label_encoders

# 准备特征
def prepare_features(df_processed):
    # 选择特征
    feature_cols = ['age', 'gender', 'income_level', 'education_level',
                   'marital_status', 'family_members', 'premium_amount',
                   'claim_history']
    
    X = df_processed[feature_cols]
    y = df_processed['renewal']
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, y, feature_cols

# 训练逻辑回归模型
def train_model(X, y):
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # 打印模型性能
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"训练集准确率: {train_score:.4f}")
    print(f"测试集准确率: {test_score:.4f}")
    
    return model

# 可视化特征重要性
def visualize_coefficients(model, feature_names):
    # 获取系数
    coefficients = model.coef_[0]
    
    # 创建系数数据框
    coef_df = pd.DataFrame({
        '特征': feature_names,
        '系数': coefficients
    })
    
    # 按系数绝对值排序
    coef_df = coef_df.reindex(coef_df.系数.abs().sort_values(ascending=True).index)
    
    # 创建水平条形图
    plt.figure(figsize=(10, 6))
    colors = ['red' if c < 0 else 'blue' for c in coef_df['系数']]
    plt.barh(coef_df['特征'], coef_df['系数'], color=colors)
    plt.title('逻辑回归系数可视化')
    plt.xlabel('系数值')
    
    # 添加垂直线表示零点
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(script_dir, 'logistic_regression_coefficients.png'))
    plt.close()
    
    # 打印系数
    print("\n特征系数：")
    for idx, row in coef_df.iterrows():
        print(f"{row['特征']}: {row['系数']:.4f}")

# 保存模型和相关转换器
def save_model(model, label_encoders, scaler, feature_cols):
    import joblib
    
    # 创建模型目录
    model_dir = os.path.join(script_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型和相关转换器
    joblib.dump(model, os.path.join(model_dir, 'lr_model.pkl'))
    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))

# 加载模型和相关转换器
def load_model():
    import joblib
    
    model_dir = os.path.join(script_dir, 'model')
    model = joblib.load(os.path.join(model_dir, 'lr_model.pkl'))
    label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
    
    return model, label_encoders, scaler, feature_cols

# 预测新数据
def predict_new_data(test_file):
    # 加载模型和转换器
    model, label_encoders, scaler, feature_cols = load_model()
    
    # 读取测试数据
    df_test = pd.read_excel(test_file)
    
    # 预处理测试数据
    df_test_processed = df_test.copy()
    
    # 对类别变量进行编码
    categorical_cols = ['gender', 'birth_region', 'insurance_region', 'income_level',
                       'education_level', 'occupation', 'marital_status',
                       'policy_type', 'claim_history']
    
    for col in categorical_cols:
        if col in feature_cols:
            df_test_processed[col] = label_encoders[col].transform(df_test_processed[col])
    
    # 准备特征
    X_test = df_test_processed[feature_cols]
    
    # 标准化特征
    X_test_scaled = scaler.transform(X_test)
    
    # 进行预测
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    
    # 添加预测结果到原始数据
    df_test['预测续保'] = ['是' if p == 1 else '否' for p in predictions]
    df_test['续保概率'] = probabilities
    
    # 保存预测结果
    output_file = os.path.join(script_dir, 'prediction_results.xlsx')
    df_test.to_excel(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")
    
    return df_test

def main():
    # 数据预处理
    df_processed, label_encoders = preprocess_data(df)
    
    # 准备特征
    X, y, feature_cols = prepare_features(df_processed)
    
    # 获取标准化器
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练模型
    model = train_model(pd.DataFrame(X_scaled, columns=feature_cols), y)
    
    # 可视化系数
    visualize_coefficients(model, feature_cols)
    
    # 保存模型
    save_model(model, label_encoders, scaler, feature_cols)
    
    # 预测测试数据
    test_file = os.path.join(script_dir, 'policy_test.xlsx')
    if os.path.exists(test_file):
        predict_new_data(test_file)

if __name__ == '__main__':
    main()