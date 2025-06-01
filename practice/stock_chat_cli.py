import os
import sys
import json
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm'))

from llm.dashscope_client import DashscopeClient
from inventory_api import InventoryAPI

class StockChatCLI:
    def __init__(self):
        self.llm_client = None
        self.inventory_api = None
        self.messages = []
        self.init_system_message()
        
    def init_system_message(self):
        """初始化系统消息"""
        self.messages.append({
            "role": "system",
            "content": "你是一个库存查询助手，能够帮助用户查询库存信息。"
        })
        
    def init_clients(self, api_key):
        """初始化客户端"""
        try:
            self.llm_client = DashscopeClient(api_key=api_key)
            self.inventory_api = InventoryAPI()
            print("✅ 客户端初始化成功！")
            return True
        except Exception as e:
            print(f"❌ 初始化失败: {str(e)}")
            return False
            
    def process_function_call(self, function_call_obj):
        """处理function call"""
        try:
            function_name = function_call_obj.get('name')
            arguments = json.loads(function_call_obj.get('arguments', '{}'))
            
            print(f"🔧 正在调用函数: {function_name}")
            print(f"📝 参数: {arguments}")
            
            if function_name == 'get_overview_inventory':
                result = self.inventory_api.get_overview_inventory(**arguments)
            elif function_name == 'get_physical_inventory':
                result = self.inventory_api.get_physical_inventory(**arguments)
            else:
                result = {"success": False, "error": f"未知的函数: {function_name}"}
                
            return result
        except Exception as e:
            return {"success": False, "error": f"函数调用错误: {str(e)}"}
            
    def format_inventory_result(self, result, function_name):
        """格式化库存查询结果"""
        if not result.get('success', False):
            return f"❌ 查询失败: {result.get('error', '未知错误')}"
            
        data = result.get('data', [])
        if not data:
            return "📦 未找到匹配的库存记录"
            
        if function_name == 'get_overview_inventory':
            return self.format_overview_result(data)
        else:
            return self.format_physical_result(data)
            
    def format_overview_result(self, data):
        """格式化总览库存结果"""
        result = "\n📊 库存总览查询结果\n" + "="*50 + "\n"
        
        for item in data:
            result += f"\n🏷️  货品编码: {item['skuCode']}\n"
            result += f"   质量状态: {item['quality']}\n"
            result += f"   总库存: {item['qty']}\n"
            result += f"   占用数量: {item['occupied']}\n"
            result += f"   可用数量: {int(item['qty']) - item['occupied']}\n"
            result += "-" * 30 + "\n"
            
        return result
        
    def format_physical_result(self, data):
        """格式化物理库存结果"""
        result = "\n🏪 物理库存查询结果\n" + "="*50 + "\n"
        
        for item in data:
            result += f"\n🏷️  货品: {item['skuCode']} - {item.get('skuName', '')}\n"
            result += f"   储位: {item['binCode']}\n"
            result += f"   库区: {item['areaName']} ({item['areaCode']})\n"
            result += f"   质量状态: {item['quality']}\n"
            result += f"   库存数量: {item['qty']}\n"
            result += f"   占用数量: {item['occupied']}\n"
            
            batch_info = item.get('batchInfo', {})
            if batch_info:
                result += f"   生产日期: {batch_info.get('produceDate', 'N/A')}\n"
                result += f"   生产批次: {batch_info.get('productionNo', 'N/A')}\n"
            result += "-" * 30 + "\n"
            
        return result
        
    def send_message(self, user_input):
        """发送消息并处理响应"""
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})
        
        try:
            print("🤖 正在思考...")
            
            # 获取LLM响应
            response = self.llm_client.get_response(self.messages)
            
            if response is None:
                print("❌ 获取响应失败")
                return
                
            # 检查是否需要处理function call
            if self.llm_client.should_process_function_call(response):
                # 处理function call
                function_call_obj = self.llm_client.fetch_function_call_obj(response)
                function_result = self.process_function_call(function_call_obj)
                
                # 添加function call消息
                self.messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call_obj
                })
                
                # 添加function result消息
                self.messages.append({
                    "role": "function",
                    "name": function_call_obj.get('name'),
                    "content": json.dumps(function_result, ensure_ascii=False)
                })
                
                # 再次调用LLM获取最终响应
                print("🤖 正在整理结果...")
                final_response = self.llm_client.get_response(self.messages)
                if final_response:
                    content = self.llm_client.fetch_content(final_response)
                    if content:
                        self.messages.append({"role": "assistant", "content": content})
                        print(f"\n🤖 助手: {content}")
                        
                        # 显示格式化的结果
                        formatted_result = self.format_inventory_result(
                            function_result, 
                            function_call_obj.get('name')
                        )
                        print(formatted_result)
            else:
                # 普通响应
                content = self.llm_client.fetch_content(response)
                if content:
                    self.messages.append({"role": "assistant", "content": content})
                    print(f"\n🤖 助手: {content}")
                    
        except Exception as e:
            print(f"❌ 处理消息时出错: {str(e)}")
            
    def show_help(self):
        """显示帮助信息"""
        help_text = """
📋 智能库存助手 - 帮助信息
================================

🔧 命令:
  help    - 显示此帮助信息
  clear   - 清空对话历史
  quit    - 退出程序

📊 支持的查询类型:
  • 总览库存查询
  • 物理库存查询

💡 示例问题:
  • "查询苹果的库存情况"
  • "冷藏区有什么水果？"
  • "查看合格品的总库存"
  • "L-01-05储位有什么？"

================================
"""
        print(help_text)
        
    def clear_history(self):
        """清空对话历史"""
        self.messages = []
        self.init_system_message()
        print("🗑️ 对话历史已清空")
        
    def run(self):
        """运行命令行聊天程序"""
        print("\n" + "="*60)
        print("📦 智能库存助手 - 命令行版本")
        print("="*60)
        
        # 获取API Key
        api_key = os.getenv("DASHSCOPE_APP_KEY")
        if not api_key:
            api_key = input("请输入您的Dashscope API Key: ").strip()
            
        if not api_key:
            print("❌ 未提供API Key，程序退出")
            return
            
        # 初始化客户端
        if not self.init_clients(api_key):
            return
            
        print("\n💡 输入 'help' 查看帮助，输入 'quit' 退出程序\n")
        
        # 主循环
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if not user_input:
                    continue
                    
                # 处理命令
                if user_input.lower() == 'quit':
                    print("👋 再见！")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                    
                # 处理普通消息
                self.send_message(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    app = StockChatCLI()
    app.run()