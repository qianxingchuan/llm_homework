import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 构建Excel文件的完整路径
excel_path = os.path.join(script_dir, 'policy_data.xlsx')

# 读取数据
df = pd.read_excel(excel_path)

# 创建保存图表的目录（使用绝对路径）
eda_results_dir = os.path.join(script_dir, 'eda_results')
if not os.path.exists(eda_results_dir):
    os.makedirs(eda_results_dir)

# 1. 基本统计信息
def basic_statistics(df):
    # 数值型变量的统计描述
    numeric_stats = df.describe()
    numeric_stats.to_csv(os.path.join(eda_results_dir, 'numeric_statistics.csv'))
    
    # 类别型变量的统计描述
    categorical_cols = ['gender', 'birth_region', 'insurance_region', 'income_level', 
                       'education_level', 'occupation', 'marital_status', 
                       'policy_type', 'claim_history', 'renewal']
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        plt.figure(figsize=(10, 6))
        value_counts.plot(kind='bar')
        plt.title(f'{col}分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_results_dir, f'{col}_distribution.png'))
        plt.close()

# 2. 年龄分布分析
def age_analysis(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', bins=30)
    plt.title('年龄分布')
    plt.savefig(os.path.join(eda_results_dir, 'age_distribution.png'))
    plt.close()
    
    # 年龄与续保关系
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='renewal', y='age')
    plt.title('续保状态下的年龄分布')
    plt.savefig(os.path.join(eda_results_dir, 'age_renewal_relationship.png'))
    plt.close()

# 3. 保费金额分析
def premium_analysis(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='premium_amount', bins=30)
    plt.title('保费金额分布')
    plt.savefig(os.path.join(eda_results_dir, 'premium_distribution.png'))
    plt.close()
    
    # 保费与收入水平的关系
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='income_level', y='premium_amount')
    plt.title('收入水平与保费金额关系')
    plt.savefig(os.path.join(eda_results_dir, 'income_premium_relationship.png'))
    plt.close()

# 4. 续保率分析
def renewal_analysis(df):
    # 不同特征下的续保率
    features = ['gender', 'income_level', 'education_level', 'marital_status', 'claim_history']
    for feature in features:
        renewal_rate = df.groupby(feature)['renewal'].apply(lambda x: (x == 'Yes').mean())
        plt.figure(figsize=(10, 6))
        renewal_rate.plot(kind='bar')
        plt.title(f'{feature}与续保率关系')
        plt.ylabel('续保率')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_results_dir, f'{feature}_renewal_rate.png'))
        plt.close()

# 5. 相关性分析
def correlation_analysis(df):
    # 将类别变量转换为数值
    df_encoded = df.copy()
    categorical_cols = ['gender', 'income_level', 'education_level', 'marital_status', 'claim_history', 'renewal']
    for col in categorical_cols:
        df_encoded[col] = pd.factorize(df_encoded[col])[0]
    
    # 计算相关性
    numeric_cols = ['age', 'family_members', 'premium_amount'] + categorical_cols
    corr_matrix = df_encoded[numeric_cols].corr()
    
    # 绘制相关性热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('特征相关性分析')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_results_dir, 'correlation_heatmap.png'))
    plt.close()

# 执行所有分析
def run_eda():
    print("开始进行探索性数据分析...")
    basic_statistics(df)
    print("基本统计分析完成")
    age_analysis(df)
    print("年龄分析完成")
    premium_analysis(df)
    print("保费分析完成")
    renewal_analysis(df)
    print("续保率分析完成")
    correlation_analysis(df)
    print("相关性分析完成")
    print(f"\n所有分析已完成，结果保存在 {eda_results_dir} 目录下")

if __name__ == '__main__':
    run_eda()