import pandas as pd
import os
from datetime import datetime

class InventoryAPI:
    def __init__(self, excel_path=None):
        """初始化库存API"""
        if excel_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            excel_path = os.path.join(current_dir, '水果库存模拟数据.xlsx')
        
        self.excel_path = excel_path
        self.load_data()
    
    def load_data(self):
        """加载Excel数据"""
        try:
            self.overview_data = pd.read_excel(self.excel_path, sheet_name='总览库存')
            self.physical_data = pd.read_excel(self.excel_path, sheet_name='物理库存')
            print(f"数据加载成功: 总览库存 {len(self.overview_data)} 条记录, 物理库存 {len(self.physical_data)} 条记录")
        except Exception as e:
            print(f"加载数据失败: {e}")
            # 创建示例数据结构
            self.overview_data = pd.DataFrame({
                'skuCode': ['APPLE001', 'BANANA001', 'ORANGE001'],
                'skuName': ['红富士苹果', '香蕉', '脐橙'],
                'quality': ['GENIUNE', 'GENIUNE', 'DEFECTIVE'],
                'qty': [100, 200, 50],
                'occupied': [20, 30, 10]
            })
            self.physical_data = pd.DataFrame({
                'skuCode': ['APPLE001', 'APPLE001', 'BANANA001'],
                'skuName': ['红富士苹果', '红富士苹果', '香蕉'],
                'quality': ['GENIUNE', 'GENIUNE', 'GENIUNE'],
                'binCode': ['L-01-05', 'L-01-06', 'L-02-01'],
                'areaCode': ['N01', 'N01', 'N02'],
                'areaName': ['冷藏区A', '冷藏区A', '冷藏区B'],
                'produceDate': ['2024-01-15', '2024-01-16', '2024-01-20'],
                'productionNo': ['D123456', 'D123457', 'D123458'],
                'qty': [50, 50, 200],
                'occupied': [10, 10, 30]
            })
    
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
    
    def get_overview_inventory_by_sku(self, sku_code):
        """根据SKU编码查询总览库存
        
        Args:
            sku_code (str): SKU编码
        
        Returns:
            dict: 包含库存数量和占用数量的响应
        """
        try:
            df = self.overview_data.copy()
            
            # 精确匹配SKU编码
            df =df[(df['skuCode'] == sku_code) | (df['skuName'] == sku_code)]
            
            if df.empty:
                return {
                    "success": True,
                    "data":{
                        "skuCode": sku_code,
                        "skuName": sku_code,
                        "totalQty": 0,
                        "totalOccupied": 0,
                        "availableQty":0,
                    }
                }
            
            # 聚合同一SKU的库存数量和占用数量
            total_qty = df['qty'].sum()
            total_occupied = df['occupied'].sum()
            sku_name = df['skuName'].iloc[0] if not df.empty else ""
            
            return {
                "success": True,
                "data": {
                    "skuCode": sku_code,
                    "skuName": sku_name,
                    "totalQty": int(total_qty),
                    "totalOccupied": int(total_occupied),
                    "availableQty": int(total_qty - total_occupied)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    def get_physical_inventory_by_sku_bin(self, sku_code, bin_code=None):
        """根据SKU编码和储位编码查询物理库存
        
        Args:
            sku_code (str): SKU编码
            bin_code (str, optional): 储位编码，如果不提供则返回该SKU的所有储位信息
        
        Returns:
            dict: 物理库存详细信息
        """
        try:
            df = self.physical_data.copy()
            
            # 按SKU编码过滤
            df = df[df['skuCode'] == sku_code]
            
            if df.empty:
                return {
                    "success": False,
                    "error": f"未找到SKU编码为 {sku_code} 的物理库存",
                    "data": []
                }
            
            # 如果提供了储位编码，进一步过滤
            if bin_code:
                df = df[df['binCode'] == bin_code]
                if df.empty:
                    return {
                        "success": False,
                        "error": f"未找到SKU编码为 {sku_code}，储位编码为 {bin_code} 的物理库存",
                        "data": []
                    }
            
            # 转换为响应格式
            data = []
            for _, row in df.iterrows():
                data.append({
                    "skuCode": row['skuCode'],
                    "skuName": row['skuName'],
                    "quality": row['quality'],
                    "binCode": row['binCode'],
                    "areaCode": row['areaCode'],
                    "areaName": row['areaName'],
                    "batchInfo": {
                        "produceDate": str(row['produceDate']),
                        "productionNo": str(row['productionNo'])
                    },
                    "qty": int(row['qty']),
                    "occupied": int(row['occupied']),
                    "available": int(row['qty'] - row['occupied'])
                })
            
            return {
                "success": True,
                "data": data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    def get_overview_inventory(self, search_sku=None, quality=None):
        """总览库存查询接口（保持原有功能）
        
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
                    "skuName": row['skuName'],
                    "quality": row['quality'],
                    "qty": int(row['qty']),
                    "occupied": int(row['occupied'])
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
        """物理库存查询接口（保持原有功能）
        
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
                    "skuName": row['skuName'],
                    "quality": row['quality'],
                    "binCode": row['binCode'],
                    "areaCode": row['areaCode'],
                    "areaName": row['areaName'],
                    "batchInfo": {
                        "produceDate": str(row['produceDate']),
                        "productionNo": str(row['productionNo'])
                    },
                    "qty": int(row['qty']),
                    "occupied": int(row['occupied'])
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
    
    def get_tools(self):
        """获取库存API的工具定义，用于大模型function call
        
        Returns:
            list: 工具定义列表
        """
        return [
            {
                "name": "get_overview_inventory_by_sku",
                "description": "根据SKU编码查询总览库存信息，返回该SKU的总库存量、占用量和可用量",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku_code": {
                            "type": "string", 
                            "description": "SKU编码，例如：'APPLE001'"
                        }
                    },
                    "required": ["sku_code"]
                }
            },
            {
                "name": "get_physical_inventory_by_sku_bin",
                "description": "根据SKU编码和储位编码查询物理库存详细信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku_code": {
                            "type": "string",
                            "description": "SKU编码，例如：'APPLE001'"
                        },
                        "bin_code": {
                            "type": "string",
                            "description": "储位编码，例如：'L-01-05'。如果不提供，返回该SKU的所有储位信息"
                        }
                    },
                    "required": ["sku_code"]
                }
            },
            {
                "name": "get_overview_inventory",
                "description": "查询总览库存信息，按SKU和质量状态聚合显示库存数据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_sku": {
                            "type": "string", 
                            "description": "货品名称或编码，支持模糊搜索，例如：'苹果'、'APPLE'"
                        },
                        "quality": {
                            "type": "string", 
                            "description": "质量状态，支持自然语言或枚举值。自然语言：'合格'、'残次'、'待检'；枚举值：'GENIUNE'、'DEFECTIVE'、'GRADE'"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_physical_inventory", 
                "description": "查询物理库存信息，显示详细的储位、批次等信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_sku": {
                            "type": "string",
                            "description": "货品名称或编码，支持模糊搜索，例如：'苹果'、'APPLE'"
                        },
                        "quality": {
                            "type": "string",
                            "description": "质量状态，支持自然语言或枚举值。自然语言：'合格'、'残次'、'待检'；枚举值：'GENIUNE'、'DEFECTIVE'、'GRADE'"
                        },
                        "bin_code": {
                            "type": "string",
                            "description": "储位编码，精确匹配，例如：'L-01-05'"
                        },
                        "area": {
                            "type": "string", 
                            "description": "库区编码或名称，支持模糊搜索，例如：'N01'、'冷藏'"
                        },
                        "batch_info": {
                            "type": "object",
                            "description": "批次信息对象",
                            "properties": {
                                "produceDate": {
                                    "type": "string",
                                    "description": "生产日期，格式：YYYY-MM-DD"
                                },
                                "productionNo": {
                                    "type": "string", 
                                    "description": "生产批次号，例如：'D123456'"
                                }
                            }
                        }
                    },
                    "required": []
                }
            }
        ]

# 使用示例
if __name__ == "__main__":
    # 初始化API
    api = InventoryAPI()
    
    print("=== 库存API测试 ===")
    
    # 首先查看实际数据中的SKU
    print("\n实际数据中的SKU列表:")
    if not api.overview_data.empty:
        unique_skus = api.overview_data['skuCode'].unique()
        print(f"总览库存中的SKU: {list(unique_skus)}")
    
    if not api.physical_data.empty:
        unique_skus_physical = api.physical_data['skuCode'].unique()
        print(f"物理库存中的SKU: {list(unique_skus_physical)}")
        
        # 获取第一个SKU用于测试
        test_sku = unique_skus_physical[0] if len(unique_skus_physical) > 0 else None
        
        if test_sku:
            # 测试1: 根据SKU查询总览库存
            print(f"\n1. 根据SKU查询总览库存 ({test_sku}):")
            result1 = api.get_overview_inventory_by_sku(test_sku)
            print(f"查询{test_sku}总览库存: {result1}")
            
            # 获取该SKU的储位信息用于测试
            sku_bins = api.physical_data[api.physical_data['skuCode'] == test_sku]['binCode'].unique()
            test_bin = sku_bins[0] if len(sku_bins) > 0 else None
            
            if test_bin:
                # 测试2: 根据SKU+储位查询物理库存
                print(f"\n2. 根据SKU+储位查询物理库存 ({test_sku} + {test_bin}):")
                result2 = api.get_physical_inventory_by_sku_bin(test_sku, test_bin)
                print(f"查询{test_sku}在{test_bin}的物理库存: {result2}")
            
            # 测试3: 根据SKU查询所有物理库存
            print(f"\n3. 根据SKU查询所有物理库存 ({test_sku}):")
            result3 = api.get_physical_inventory_by_sku_bin(test_sku)
            print(f"查询{test_sku}所有物理库存: {result3}")
    
    # 测试4: 查询不存在的SKU
    print("\n4. 查询不存在的SKU:")
    result4 = api.get_overview_inventory_by_sku("NOTEXIST001")
    print(f"查询不存在的SKU: {result4}")
    
    # 测试5: 展示更多功能
    print("\n5. 查询所有总览库存:")
    result5 = api.get_overview_inventory()
    print(f"所有总览库存: {result5}")
    
    print("\n6. 查询所有物理库存:")
    result6 = api.get_physical_inventory()
    print(f"所有物理库存: {result6}")
    
    # 获取工具定义
    print("\n=== 工具定义 ===")
    tools = api.get_tools()
    import json
    print(json.dumps(tools, ensure_ascii=False, indent=2))