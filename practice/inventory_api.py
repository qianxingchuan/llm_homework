import pandas as pd
import os
from datetime import datetime

class InventoryAPI:
    def __init__(self, excel_path=None):
        """初始化库存API"""
        if excel_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            excel_path = os.path.join(current_dir, 'smart_stock', 'mock_data', '水果库存模拟数据.xlsx')
        
        self.excel_path = excel_path
        self.load_data()
    
    def load_data(self):
        """加载Excel数据"""
        try:
            self.overview_data = pd.read_excel(self.excel_path, sheet_name='总览库存')
            self.physical_data = pd.read_excel(self.excel_path, sheet_name='物理库存')
        except Exception as e:
            print(f"加载数据失败: {e}")
            self.overview_data = pd.DataFrame()
            self.physical_data = pd.DataFrame()
    
    def _normalize_quality(self, quality_input):
        """将自然语言质量状态转换为枚举值"""
        if not quality_input:
            return None
        
        quality_map = {
            '合格': 'GENIUNE',
            '残次': 'DEFECTIVE', 
            '待检': 'GRADE',
            'GENIUNE': 'GENIUNE',
            'DEFECTIVE': 'DEFECTIVE',
            'GRADE': 'GRADE'
        }
        
        return quality_map.get(quality_input, quality_input)
    
    def get_overview_inventory(self, search_sku=None, quality=None):
        """总览库存查询接口
        
        Args:
            search_sku (str, optional): 货品名称或编码，支持模糊搜索
            quality (str, optional): 质量状态，支持自然语言
        
        Returns:
            dict: API响应格式
        """
        try:
            df = self.overview_data.copy()
            
            # 按searchSku过滤（支持名称和编码模糊搜索）
            if search_sku:
                mask = (df['skuName'].str.contains(search_sku, na=False, case=False) | 
                       df['skuCode'].str.contains(search_sku, na=False, case=False))
                df = df[mask]
            
            # 按quality过滤
            if quality:
                normalized_quality = self._normalize_quality(quality)
                if normalized_quality:
                    df = df[df['quality'] == normalized_quality]
            
            # 转换为响应格式
            data = []
            for _, row in df.iterrows():
                data.append({
                    "skuCode": row['skuCode'],
                    "quality": row['quality'],
                    "qty": str(row['qty']),
                    "occupied": row['occupied']
                })
            
            return {
                "success": True,
                "data": data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_physical_inventory(self, search_sku=None, quality=None, bin_code=None, 
                             area=None, batch_info=None):
        """物理库存查询接口
        
        Args:
            search_sku (str, optional): 货品名称或编码，支持模糊搜索
            quality (str, optional): 质量状态，支持自然语言
            bin_code (str, optional): 储位编码
            area (str, optional): 库区编码或名称，支持模糊搜索
            batch_info (dict, optional): 批次信息，包含produceDate和productionNo
        
        Returns:
            dict: API响应格式
        """
        try:
            df = self.physical_data.copy()
            
            # 按searchSku过滤（支持名称和编码模糊搜索）
            if search_sku:
                mask = (df['skuName'].str.contains(search_sku, na=False, case=False) | 
                       df['skuCode'].str.contains(search_sku, na=False, case=False))
                df = df[mask]
            
            # 按quality过滤
            if quality:
                normalized_quality = self._normalize_quality(quality)
                if normalized_quality:
                    df = df[df['quality'] == normalized_quality]
            
            # 按binCode过滤
            if bin_code:
                df = df[df['binCode'] == bin_code]
            
            # 按area过滤（支持编码和名称模糊搜索）
            if area:
                mask = (df['areaCode'].str.contains(area, na=False, case=False) | 
                       df['areaName'].str.contains(area, na=False, case=False))
                df = df[mask]
            
            # 按批次信息过滤
            if batch_info:
                if 'produceDate' in batch_info and batch_info['produceDate']:
                    df = df[df['produceDate'] == batch_info['produceDate']]
                if 'productionNo' in batch_info and batch_info['productionNo']:
                    df = df[df['productionNo'] == batch_info['productionNo']]
            
            # 转换为响应格式
            data = []
            for _, row in df.iterrows():
                data.append({
                    "skuCode": row['skuCode'],
                    "quality": row['quality'],
                    "binCode": row['binCode'],
                    "areaCode": row['areaCode'],
                    "areaName": row['areaName'],
                    "batchInfo": {
                        "produceDate": row['produceDate'],
                        "productionNo": row['productionNo']
                    },
                    "qty": str(row['qty']),
                    "occupied": row['occupied']
                })
            
            return {
                "success": True,
                "data": data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# 使用示例
if __name__ == "__main__":
    # 初始化API
    api = InventoryAPI()
    
    print("=== 总览库存查询示例 ===")
    
    # 示例1: 查询所有苹果的库存概览
    result1 = api.get_overview_inventory(search_sku="苹果", quality="合格")
    print("查询苹果合格品库存概览:")
    print(result1)
    print()
    
    # 示例2: 查询所有库存概览
    result2 = api.get_overview_inventory()
    print(f"查询所有库存概览，共{len(result2['data'])}条记录")
    print()
    
    print("=== 物理库存查询示例 ===")
    
    # 示例3: 查询苹果的物理库存
    result3 = api.get_physical_inventory(search_sku="苹果")
    print("查询苹果物理库存:")
    print(result3)
    print()
    
    # 示例4: 按库区查询
    result4 = api.get_physical_inventory(area="冷藏")
    print("查询冷藏区库存:")
    print(f"找到{len(result4['data'])}条记录")
    print()
    
    # 示例5: 按储位查询
    if result3['data']:
        bin_code = result3['data'][0]['binCode']
        result5 = api.get_physical_inventory(bin_code=bin_code)
        print(f"查询储位{bin_code}的库存:")
        print(result5)