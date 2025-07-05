import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def calculate_mae_from_submission(submission_file):
    """从提交文件计算MAE"""
    # 读取提交文件
    submission = pd.read_csv(submission_file)
    
    # 读取测试集真实值（如果有的话）
    try:
        test_data = pd.read_csv('used_car_testB_20200421.csv')
        if 'price' in test_data.columns:
            true_prices = test_data['price'].values
            pred_prices = submission['price'].values
            
            mae = mean_absolute_error(true_prices, pred_prices)
            print(f"真实MAE: {mae:.2f}")
            return mae
        else:
            print("测试集中没有price列，无法计算真实MAE")
            return None
    except:
        print("无法读取测试集文件")
        return None

def analyze_predictions(submission_file):
    """分析预测结果"""
    submission = pd.read_csv(submission_file)
    prices = submission['price'].values
    
    print(f"预测结果统计:")
    print(f"  样本数量: {len(prices)}")
    print(f"  价格范围: {prices.min():.2f} - {prices.max():.2f}")
    print(f"  平均价格: {prices.mean():.2f}")
    print(f"  中位数价格: {np.median(prices):.2f}")
    print(f"  标准差: {prices.std():.2f}")
    
    # 检查异常值
    q1 = np.percentile(prices, 25)
    q3 = np.percentile(prices, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
    print(f"  异常值数量: {len(outliers)} ({len(outliers)/len(prices)*100:.2f}%)")
    
    if len(outliers) > 0:
        print(f"  异常值范围: {outliers.min():.2f} - {outliers.max():.2f}")

def compare_submissions():
    """比较不同的提交文件"""
    files_to_check = [
        ('ensemble_submission.csv', '集成方法'),
        ('gpu_accelerated_submission_fixed.csv', 'GPU训练方法')
    ]
    
    print("=== 比较不同方法的预测结果 ===")
    
    results = {}
    for file, name in files_to_check:
        try:
            print(f"\n{name} ({file}):")
            analyze_predictions(file)
            mae = calculate_mae_from_submission(file)
            if mae:
                results[name] = mae
                print(f"MAE: {mae:.2f}")
        except FileNotFoundError:
            print(f"文件 {file} 不存在")
    
    # 比较结果
    if len(results) >= 2:
        print(f"\n=== 结果对比 ===")
        for name, mae in results.items():
            print(f"{name}: {mae:.2f}")
        
        best_method = min(results, key=results.get)
        worst_method = max(results, key=results.get)
        improvement = results[worst_method] - results[best_method]
        
        print(f"\n最佳方法: {best_method} (MAE: {results[best_method]:.2f})")
        print(f"改进幅度: {improvement:.2f}")

if __name__ == "__main__":
    # 比较所有方法
    compare_submissions() 