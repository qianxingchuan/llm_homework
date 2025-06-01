import streamlit as st
import os
import sys
import json
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm'))

from llm.dashscope_client import DashscopeClient
from inventory_api import InventoryAPI

class StockChatApp:
    def __init__(self):
        self.llm_client = None
        self.inventory_api = None
        self.init_session_state()
        
    def init_session_state(self):
        """初始化session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "system",
                "content": "你是一个库存查询助手，能够帮助用户查询库存信息。"
            })
        if 'api_key' not in st.session_state:
            st.session_state.api_key = os.getenv("DASHSCOPE_APP_KEY", "")
        if 'client_initialized' not in st.session_state:
            st.session_state.client_initialized = False
            
    def init_clients(self, api_key):
        """初始化客户端"""
        try:
            self.llm_client = DashscopeClient(api_key=api_key)
            self.inventory_api = InventoryAPI()
            st.session_state.client_initialized = True
            return True
        except Exception as e:
            st.error(f"初始化失败: {str(e)}")
            return False
            
    def process_function_call(self, function_call_obj):
        """处理function call"""
        try:
            function_name = function_call_obj.get('name')
            arguments = json.loads(function_call_obj.get('arguments', '{}'))
            
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
        result = "📊 **库存总览查询结果**\n\n"
        
        for item in data:
            result += f"**{item['skuCode']}**\n"
            result += f"- 质量状态: {item['quality']}\n"
            result += f"- 总库存: {item['qty']}\n"
            result += f"- 占用数量: {item['occupied']}\n"
            result += f"- 可用数量: {int(item['qty']) - item['occupied']}\n\n"
            
        return result
        
    def format_physical_result(self, data):
        """格式化物理库存结果"""
        result = "🏪 **物理库存查询结果**\n\n"
        
        for item in data:
            result += f"**{item['skuCode']} - {item.get('skuName', '')}**\n"
            result += f"- 储位: {item['binCode']}\n"
            result += f"- 库区: {item['areaName']} ({item['areaCode']})\n"
            result += f"- 质量状态: {item['quality']}\n"
            result += f"- 库存数量: {item['qty']}\n"
            result += f"- 占用数量: {item['occupied']}\n"
            
            batch_info = item.get('batchInfo', {})
            if batch_info:
                result += f"- 生产日期: {batch_info.get('produceDate', 'N/A')}\n"
                result += f"- 生产批次: {batch_info.get('productionNo', 'N/A')}\n"
            result += "\n"
            
        return result
        
    def send_message(self, user_input):
        """发送消息并处理响应"""
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            # 获取LLM响应
            response = self.llm_client.get_response(st.session_state.messages)
            
            if response is None:
                st.error("获取响应失败")
                return
                
            # 检查是否需要处理function call
            if self.llm_client.should_process_function_call(response):
                # 处理function call
                function_call_obj = self.llm_client.fetch_function_call_obj(response)
                function_result = self.process_function_call(function_call_obj)
                
                # 添加function call消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call_obj
                })
                
                # 添加function result消息
                st.session_state.messages.append({
                    "role": "function",
                    "name": function_call_obj.get('name'),
                    "content": json.dumps(function_result, ensure_ascii=False)
                })
                
                # 再次调用LLM获取最终响应
                final_response = self.llm_client.get_response(st.session_state.messages)
                if final_response:
                    content = self.llm_client.fetch_content(final_response)
                    if content:
                        st.session_state.messages.append({"role": "assistant", "content": content})
                        
                        # 如果是库存查询，添加格式化的结果展示
                        formatted_result = self.format_inventory_result(
                            function_result, 
                            function_call_obj.get('name')
                        )
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": formatted_result,
                            "type": "formatted_result"
                        })
            else:
                # 普通响应
                content = self.llm_client.fetch_content(response)
                if content:
                    st.session_state.messages.append({"role": "assistant", "content": content})
                    
        except Exception as e:
            st.error(f"处理消息时出错: {str(e)}")
            
    def render_message(self, message):
        """渲染单条消息"""
        role = message["role"]
        content = message.get("content", "")
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        elif role == "assistant" and content:
            with st.chat_message("assistant"):
                if message.get("type") == "formatted_result":
                    st.markdown(content)
                else:
                    st.write(content)
        elif role == "function":
            # 不显示function消息，只在调试模式下显示
            if st.session_state.get('debug_mode', False):
                with st.expander(f"🔧 函数调用: {message.get('name', 'unknown')}"):
                    st.json(json.loads(content))
                    
    def run(self):
        """运行应用"""
        st.set_page_config(
            page_title="智能库存助手",
            page_icon="📦",
            layout="wide"
        )
        
        st.title("📦 智能库存助手")
        st.markdown("---")
        
        # 侧边栏配置
        with st.sidebar:
            st.header("⚙️ 配置")
            
            # API Key输入
            api_key = st.text_input(
                "Dashscope API Key", 
                value=st.session_state.api_key,
                type="password",
                help="请输入您的Dashscope API Key"
            )
            
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                st.session_state.client_initialized = False
                
            # 初始化按钮
            if st.button("🔄 初始化客户端", type="primary"):
                if api_key:
                    if self.init_clients(api_key):
                        st.success("✅ 客户端初始化成功！")
                    else:
                        st.error("❌ 客户端初始化失败")
                else:
                    st.error("请先输入API Key")
                    
            # 调试模式
            st.session_state.debug_mode = st.checkbox("🐛 调试模式")
            
            # 清空对话
            if st.button("🗑️ 清空对话"):
                st.session_state.messages = []
                st.rerun()
                
            st.markdown("---")
            st.markdown("### 📋 功能说明")
            st.markdown("""
            **支持的查询类型：**
            - 📊 总览库存查询
            - 🏪 物理库存查询
            
            **示例问题：**
            - "查询苹果的库存情况"
            - "冷藏区有什么水果？"
            - "查看合格品的总库存"
            - "L-01-05储位有什么？"
            """)
            
        # 主聊天区域
        if not st.session_state.client_initialized:
            st.warning("⚠️ 请先在侧边栏配置API Key并初始化客户端")
            return
            
        # 显示对话历史
        for message in st.session_state.messages:
            self.render_message(message)
            
        # 用户输入
        if prompt := st.chat_input("请输入您的问题..."):
            self.send_message(prompt)
            st.rerun()
            
        # 快捷查询按钮
        st.markdown("### 🚀 快捷查询")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🍎 查询苹果库存"):
                self.send_message("查询苹果的库存情况")
                st.rerun()
                
        with col2:
            if st.button("❄️ 查询冷藏区库存"):
                self.send_message("冷藏区有什么水果？")
                st.rerun()
                
        with col3:
            if st.button("✅ 查询合格品库存"):
                self.send_message("查看所有合格品的库存")
                st.rerun()
                
        with col4:
            if st.button("📈 库存总览"):
                self.send_message("显示所有水果的库存总览")
                st.rerun()

if __name__ == "__main__":
    app = StockChatApp()
    app.run()

