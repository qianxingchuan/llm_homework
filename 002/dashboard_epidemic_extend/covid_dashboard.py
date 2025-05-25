from flask import Flask, render_template, jsonify
import pandas as pd
import json
import os

app = Flask(__name__)

# 读取数据
def load_data():
    file_path = "香港各区疫情数据_20250322.xlsx"
    df = pd.read_excel(file_path)
    df['报告日期'] = pd.to_datetime(df['报告日期'])
    return df

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 每日新增和累计确诊数据API
@app.route('/api/daily_cases')
def daily_cases():
    df = load_data()
    daily_total = df.groupby('报告日期').agg({
        '新增确诊': 'sum',
        '累计确诊': 'max'
    }).reset_index()
    
    result = {
        'dates': daily_total['报告日期'].dt.strftime('%Y-%m-%d').tolist(),
        'new_cases': daily_total['新增确诊'].tolist(),
        'total_cases': daily_total['累计确诊'].tolist()
    }
    return jsonify(result)

# 地区分布数据API
@app.route('/api/district_distribution')
def district_distribution():
    df = load_data()
    latest_date = df['报告日期'].max()
    latest_data = df[df['报告日期'] == latest_date]
    
    district_data = []
    for _, row in latest_data.iterrows():
        district_data.append({
            'name': row['地区名称'],
            'value': int(row['现存确诊']),
            'total': int(row['累计确诊']),
            'population': int(row['人口']),
            'rate': float(row['发病率(每10万人)'])
        })
    
    return jsonify(district_data)

# 风险等级分布API
@app.route('/api/risk_distribution')
def risk_distribution():
    df = load_data()
    risk_counts = df['风险等级'].value_counts().reset_index()
    risk_counts.columns = ['风险等级', '记录数']
    
    result = {
        'categories': risk_counts['风险等级'].tolist(),
        'values': risk_counts['记录数'].tolist()
    }
    return jsonify(result)

# 增长率变化API
@app.route('/api/growth_rate')
def growth_rate():
    df = load_data()
    daily_total = df.groupby('报告日期').agg({
        '新增确诊': 'sum',
        '累计确诊': 'max'
    }).reset_index()
    
    # 计算7日移动平均
    daily_total['7日移动平均'] = daily_total['新增确诊'].rolling(window=7).mean()
    
    # 计算增长率
    daily_total['增长率'] = daily_total['新增确诊'].pct_change() * 100
    
    result = {
        'dates': daily_total['报告日期'].dt.strftime('%Y-%m-%d').tolist(),
        'growth_rate': daily_total['增长率'].fillna(0).tolist(),
        'moving_avg': daily_total['7日移动平均'].fillna(0).tolist()
    }
    return jsonify(result)

# 地区排名API
@app.route('/api/district_ranking')
def district_ranking():
    """
    API endpoint that returns COVID-19 district ranking data.
    
    Returns:
        JSON: A dictionary containing:
            - districts: List of district names (sorted by infection rate)
            - cases: List of total confirmed cases per district
            - rates: List of confirmed cases per 100k population for each district
    """
    df = load_data()
    district_total = df.groupby('地区名称').agg({
        '新增确诊': 'sum',
        '人口': 'first'
    }).reset_index()
    
    # 计算每10万人确诊率并保留2位小数
    district_total['每10万人确诊率'] = round((district_total['新增确诊'] / district_total['人口']) * 100000, 2)
    district_total = district_total.sort_values('每10万人确诊率', ascending=True)
    
    result = {
        'districts': district_total['地区名称'].tolist(),
        'cases': district_total['新增确诊'].tolist(),
        'rates': district_total['每10万人确诊率'].tolist()
    }
    return jsonify(result)

# 创建必要的目录
def create_directories():
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)

if __name__ == '__main__':
    create_directories()
    app.run(debug=True)