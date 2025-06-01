import pandas as pd
import random
from datetime import datetime, timedelta
import os
# 水果类型数据
fruits = [
    {"name": "苹果", "code": "APPLE"},
    {"name": "香蕉", "code": "BANANA"},
    {"name": "橙子", "code": "ORANGE"},
    {"name": "葡萄", "code": "GRAPE"},
    {"name": "草莓", "code": "STRAWBERRY"},
    {"name": "猕猴桃", "code": "KIWI"},
    {"name": "芒果", "code": "MANGO"},
    {"name": "菠萝", "code": "PINEAPPLE"},
    {"name": "西瓜", "code": "WATERMELON"},
    {"name": "梨", "code": "PEAR"}
]

# 质量状态
qualities = ["GENIUNE", "DEFECTIVE", "GRADE"]

# 库区信息
areas = [
    {"code": "N01", "name": "常规01区"},
    {"code": "N02", "name": "常规02区"},
    {"code": "C01", "name": "冷藏01区"},
    {"code": "C02", "name": "冷藏02区"},
    {"code": "F01", "name": "冷冻01区"}
]

# 生成物理库存数据
def generate_physical_inventory(num_records=30):
    data = []
    
    for i in range(num_records):
        fruit = random.choice(fruits)
        quality = random.choice(qualities)
        area = random.choice(areas)
        
        # 生成储位编码
        bin_code = f"L-{random.randint(1, 10):02d}-{random.randint(1, 20):02d}"
        
        # 生成批次信息
        produce_date = datetime.now() - timedelta(days=random.randint(1, 365))
        production_no = f"D{random.randint(100000, 999999)}"
        
        # 生成库存数量
        qty = random.randint(50, 500)
        occupied = random.randint(0, min(qty, 100))
        
        record = {
            "skuCode": fruit["code"],
            "skuName": fruit["name"],
            "quality": quality,
            "binCode": bin_code,
            "areaCode": area["code"],
            "areaName": area["name"],
            "produceDate": produce_date.strftime("%Y-%m-%d"),
            "productionNo": production_no,
            "qty": qty,
            "occupied": occupied
        }
        
        data.append(record)
    
    return data

# 生成总览库存数据（按SKU和质量聚合）
def generate_overview_inventory(physical_data):
    overview_data = []
    
    # 按skuCode和quality分组聚合
    df_physical = pd.DataFrame(physical_data)
    grouped = df_physical.groupby(['skuCode', 'skuName', 'quality']).agg({
        'qty': 'sum',
        'occupied': 'sum'
    }).reset_index()
    
    for _, row in grouped.iterrows():
        record = {
            "skuCode": row['skuCode'],
            "skuName": row['skuName'],
            "quality": row['quality'],
            "qty": int(row['qty']),
            "occupied": int(row['occupied'])
        }
        overview_data.append(record)
    
    return overview_data

# 生成数据
if __name__ == "__main__":
    # 生成物理库存数据
    physical_data = generate_physical_inventory(30)
    
    # 生成总览库存数据
    overview_data = generate_overview_inventory(physical_data)
    
    # 转换为DataFrame
    df_physical = pd.DataFrame(physical_data)
    df_overview = pd.DataFrame(overview_data)
    
    # 保存为Excel文件，包含两个工作表
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_folder = os.path.join(current_dir,'smart_stock','mock_data')
    if not os.path.exists(excel_folder):
        os.makedirs(excel_folder)
    excel_filename = "水果库存模拟数据.xlsx"
    excel_path = os.path.join(excel_folder, excel_filename)
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_physical.to_excel(writer, sheet_name='物理库存', index=False)
        df_overview.to_excel(writer, sheet_name='总览库存', index=False)
    
    print(f"已生成库存模拟数据，保存为: {excel_filename}")
    print(f"包含两个工作表：总览库存({len(df_overview)}条记录) 和 物理库存({len(df_physical)}条记录)")
    
    print("\n总览库存数据预览:")
    print(df_overview.head())
    
    print("\n物理库存数据预览:")
    print(df_physical.head())
    
    # 显示数据统计
    print("\n数据统计:")
    print(f"物理库存记录数: {len(df_physical)}")
    print(f"总览库存记录数: {len(df_overview)}")
    print(f"水果种类: {df_physical['skuName'].nunique()}")
    print(f"质量状态分布: {df_physical['quality'].value_counts().to_dict()}")
    print(f"库区分布: {df_physical['areaName'].value_counts().to_dict()}")