import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取数据
df = pd.read_excel(os.path.join(script_dir, 'policy_data.xlsx'))

print("数据基本信息：")
print(f"数据形状: {df.shape}")
print(f"\n前5行数据:")
print(df.head())

# 数据预处理函数
def preprocess_data(df):
    """
    数据预处理函数
    参数:
        df: 原始数据框
    返回:
        df_processed: 处理后的数据框
        label_encoders: 标签编码器字典
    """
    # 复制数据框，避免修改原始数据
    df_processed = df.copy()
    
    # 定义需要编码的类别变量
    categorical_cols = ['gender', 'birth_region', 'insurance_region', 'income_level',
                       'education_level', 'occupation', 'marital_status',
                       'policy_type', 'claim_history']
    
    # 创建标签编码器字典
    label_encoders = {}
    
    # 对每个类别变量进行标签编码
    for col in categorical_cols:
        if col in df_processed.columns:
            label_encoders[col] = LabelEncoder()
            df_processed[col] = label_encoders[col].fit_transform(df_processed[col])
            print(f"{col} 编码映射: {dict(zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)))}")
    
    # 将目标变量转换为数值（Yes=1, No=0）
    df_processed['renewal'] = (df_processed['renewal'] == 'Yes').astype(int)
    
    return df_processed, label_encoders

# 准备特征和目标变量
def prepare_features(df_processed):
    """
    准备特征和目标变量
    参数:
        df_processed: 预处理后的数据框
    返回:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
    """
    # 选择特征列
    feature_cols = ['age', 'gender', 'income_level', 'education_level',
                   'marital_status', 'family_members', 'premium_amount',
                   'claim_history']
    
    X = df_processed[feature_cols]
    y = df_processed['renewal']
    
    print(f"\n特征列: {feature_cols}")
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标变量分布:")
    print(y.value_counts())
    
    return X, y, feature_cols

# 训练决策树模型
def train_decision_tree(X, y, max_depth=4):
    """
    训练决策树模型
    参数:
        X: 特征矩阵
        y: 目标变量
        max_depth: 决策树最大深度
    返回:
        model: 训练好的决策树模型
        X_train, X_test, y_train, y_test: 训练集和测试集
    """
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n数据分割结果:")
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 创建决策树分类器
    model = DecisionTreeClassifier(
        max_depth=max_depth,      # 最大深度
        random_state=42,          # 随机种子
        min_samples_split=20,     # 内部节点再划分所需最小样本数
        min_samples_leaf=10,      # 叶子节点最少样本数
        criterion='gini'          # 分割标准（基尼不纯度）
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算准确率
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n模型性能:")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 打印分类报告
    print(f"\n测试集分类报告:")
    print(classification_report(y_test, y_test_pred, target_names=['不续保', '续保']))
    
    return model, X_train, X_test, y_train, y_test

# 打印决策树规则
def print_tree_rules(model, feature_names):
    """
    打印决策树的文本规则
    参数:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
    """
    print("\n=== 决策树规则 ===")
    tree_rules = export_text(model, feature_names=feature_names)
    print(tree_rules)
    
    # 保存规则到文件
    rules_file = os.path.join(script_dir, 'decision_tree_rules.txt')
    with open(rules_file, 'w', encoding='utf-8') as f:
        f.write("决策树规则\n")
        f.write("=" * 50 + "\n")
        f.write(tree_rules)
    print(f"\n决策树规则已保存到: {rules_file}")

# 可视化决策树
def visualize_decision_tree(model, feature_names):
    """
    可视化决策树
    参数:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
    """
    # 创建图形
    plt.figure(figsize=(20, 12))
    
    # 绘制决策树
    plot_tree(model, 
              feature_names=feature_names,
              class_names=['不续保', '续保'],
              filled=True,                    # 填充颜色
              rounded=True,                   # 圆角矩形
              fontsize=10,                    # 字体大小
              max_depth=3)                    # 显示的最大深度
    
    plt.title('决策树可视化 (深度=4)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    tree_image = os.path.join(script_dir, 'decision_tree_visualization.png')
    plt.savefig(tree_image, dpi=300, bbox_inches='tight')
    print(f"决策树可视化图已保存到: {tree_image}")
    plt.show()

# 特征重要性分析
def analyze_feature_importance(model, feature_names):
    """
    分析特征重要性
    参数:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
    """
    # 获取特征重要性
    importances = model.feature_importances_
    
    # 创建特征重要性数据框
    feature_importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importances
    }).sort_values('重要性', ascending=False)
    
    print("\n=== 特征重要性排序 ===")
    for idx, row in feature_importance_df.iterrows():
        print(f"{row['特征']}: {row['重要性']:.4f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='重要性', y='特征', palette='viridis')
    plt.title('决策树特征重要性', fontsize=14, fontweight='bold')
    plt.xlabel('重要性分数')
    plt.ylabel('特征')
    
    # 在条形图上添加数值标签
    for i, v in enumerate(feature_importance_df['重要性']):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    
    # 保存图片
    importance_image = os.path.join(script_dir, 'feature_importance.png')
    plt.savefig(importance_image, dpi=300, bbox_inches='tight')
    print(f"特征重要性图已保存到: {importance_image}")
    plt.show()
    
    return feature_importance_df

# 混淆矩阵可视化
def plot_confusion_matrix(y_true, y_pred):
    """
    绘制混淆矩阵
    参数:
        y_true: 真实标签
        y_pred: 预测标签
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['不续保', '续保'],
                yticklabels=['不续保', '续保'])
    plt.title('混淆矩阵', fontsize=14, fontweight='bold')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    plt.tight_layout()
    
    # 保存图片
    cm_image = os.path.join(script_dir, 'confusion_matrix.png')
    plt.savefig(cm_image, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵图已保存到: {cm_image}")
    plt.show()

# 主函数
def main():
    """
    主函数：执行完整的决策树分析流程
    """
    print("=" * 60)
    print("决策树分类预测 - 客户续保分析")
    print("=" * 60)
    
    # 1. 数据预处理
    print("\n1. 数据预处理...")
    df_processed, label_encoders = preprocess_data(df)
    
    # 2. 准备特征
    print("\n2. 准备特征和目标变量...")
    X, y, feature_names = prepare_features(df_processed)
    
    # 3. 训练决策树模型
    print("\n3. 训练决策树模型...")
    model, X_train, X_test, y_train, y_test = train_decision_tree(X, y, max_depth=4)
    
    # 4. 打印决策树规则
    print("\n4. 生成决策树规则...")
    print_tree_rules(model, feature_names)
    
    # 5. 可视化决策树
    print("\n5. 可视化决策树...")
    visualize_decision_tree(model, feature_names)
    
    # 6. 特征重要性分析
    print("\n6. 分析特征重要性...")
    feature_importance_df = analyze_feature_importance(model, feature_names)
    
    # 7. 混淆矩阵
    print("\n7. 生成混淆矩阵...")
    y_test_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_test_pred)
    
    print("\n=" * 60)
    print("分析完成！所有结果已保存到文件。")
    print("=" * 60)
    
    return model, feature_importance_df

if __name__ == '__main__':
    model, feature_importance = main()