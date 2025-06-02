import os
import sys
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
import threading

# 添加数据可视化库
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm'))

from llm.dashscope_client import DashscopeClient
from inventory_api import InventoryAPI

class StockChatGUI:
    def __init__(self):
        self.llm_client = None
        self.inventory_api = None
        self.messages = []
        self.init_system_message()
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("📦 智能库存助手")
        self.root.geometry("1200x800")  # 增大窗口以容纳图表
        self.root.minsize(1000, 700)
        
        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.setup_ui()
        
    def init_system_message(self):
        """初始化系统消息"""
        self.messages.append({
            "role": "system",
            "content": """你是一个库存查询助手，能够帮助用户查询库存信息。
            注意：涉及到库存相关的部分，必须要调用我提供的tools来获得信息。
            """
        })
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)  # 聊天区域权重更大
        main_frame.columnconfigure(2, weight=1)  # 图表区域
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="📦 智能库存助手", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # 左侧配置面板
        config_frame = ttk.LabelFrame(main_frame, text="⚙️ 配置", padding="10")
        config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        config_frame.columnconfigure(0, weight=1)
        
        # API Key输入
        ttk.Label(config_frame, text="Dashscope API Key:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.api_key_var = tk.StringVar(value=os.getenv("DASHSCOPE_APP_KEY", ""))
        self.api_key_entry = ttk.Entry(config_frame, textvariable=self.api_key_var, show="*", width=30)
        self.api_key_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 初始化按钮
        self.init_button = ttk.Button(config_frame, text="🔄 初始化客户端", command=self.init_clients)
        self.init_button.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 状态标签
        self.status_var = tk.StringVar(value="❌ 未初始化")
        self.status_label = ttk.Label(config_frame, textvariable=self.status_var)
        self.status_label.grid(row=3, column=0, sticky=tk.W, pady=(0, 10))
        
        # 分隔线
        ttk.Separator(config_frame, orient='horizontal').grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # 快捷查询按钮
        ttk.Label(config_frame, text="🚀 快捷查询:", font=('Arial', 10, 'bold')).grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        
        quick_queries = [
            ("🍎 查询苹果库存", "查询苹果的库存情况"),
            ("❄️ 查询冷藏区库存", "冷藏区有什么水果？"),
            ("✅ 查询合格品库存", "查看所有合格品的库存"),
            ("📈 库存总览", "显示所有水果的库存总览")
        ]
        
        for i, (text, query) in enumerate(quick_queries):
            btn = ttk.Button(config_frame, text=text, 
                           command=lambda q=query: self.send_quick_query(q))
            btn.grid(row=6+i, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # 分隔线
        ttk.Separator(config_frame, orient='horizontal').grid(row=10, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # 图表控制按钮
        ttk.Label(config_frame, text="📊 图表选项:", font=('Arial', 10, 'bold')).grid(row=11, column=0, sticky=tk.W, pady=(0, 5))
        
        chart_buttons = [
            ("📊 生成表格", self.show_table_chart),
            ("📈 生成折线图", self.show_line_chart),
            ("🥧 生成饼图", self.show_pie_chart),
            ("📊 生成柱状图", self.show_bar_chart)
        ]
        
        for i, (text, command) in enumerate(chart_buttons):
            btn = ttk.Button(config_frame, text=text, command=command)
            btn.grid(row=12+i, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # 分隔线
        ttk.Separator(config_frame, orient='horizontal').grid(row=16, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # 控制按钮
        ttk.Button(config_frame, text="🗑️ 清空对话", command=self.clear_history).grid(row=17, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(config_frame, text="❓ 帮助", command=self.show_help).grid(row=18, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # 中间聊天区域
        chat_frame = ttk.LabelFrame(main_frame, text="💬 对话区域", padding="10")
        chat_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # 聊天显示区域
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED, 
                                                     font=('Arial', 10), height=25)
        self.chat_display.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 配置文本标签
        self.chat_display.tag_configure("user", foreground="blue", font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure("assistant", foreground="green", font=('Arial', 10, 'bold'))
        self.chat_display.tag_configure("system", foreground="gray", font=('Arial', 9, 'italic'))
        self.chat_display.tag_configure("error", foreground="red", font=('Arial', 10, 'bold'))
        
        # 输入区域
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        input_frame.columnconfigure(0, weight=1)
        
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_var, font=('Arial', 10))
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.input_entry.bind('<Return>', self.send_message_event)
        
        self.send_button = ttk.Button(input_frame, text="发送", command=self.send_message)
        self.send_button.grid(row=0, column=1)
        
        # 右侧图表区域
        chart_frame = ttk.LabelFrame(main_frame, text="📊 数据可视化", padding="10")
        chart_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        # 图表显示区域
        self.chart_frame_inner = ttk.Frame(chart_frame)
        self.chart_frame_inner.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 存储最新的查询结果用于图表生成
        self.latest_result = None
        self.latest_function_name = None
        
        # 添加欢迎消息
        self.add_message("system", "欢迎使用智能库存助手！请先配置API Key并初始化客户端。")
        
    # ... existing code ...
    # 保持原有的所有方法不变，只需要修改 format_inventory_result 方法
    
    def format_inventory_result(self, result, function_name):
        """格式化库存查询结果并存储用于图表生成"""
        # 存储最新结果用于图表生成
        self.latest_result = result
        self.latest_function_name = function_name
        
        if not result.get('success', False):
            return f"❌ 查询失败: {result.get('error', '未知错误')}"
            
        data = result.get('data', [])
        if not data:
            return "📦 未找到匹配的库存记录"
            
        if function_name == 'get_overview_inventory':
            formatted_text = self.format_overview_result(data)
        else:
            formatted_text = self.format_physical_result(data)
            
        # 自动生成表格
        self.root.after(100, self.show_table_chart)
        
        return formatted_text
    
    def show_table_chart(self):
        """显示表格图表"""
        if not self.latest_result or not self.latest_result.get('success'):
            messagebox.showwarning("警告", "没有可用的数据生成表格")
            return
            
        data = self.latest_result.get('data', [])
        if not data:
            return
            
        # 清除之前的图表
        for widget in self.chart_frame_inner.winfo_children():
            widget.destroy()
            
        # 创建表格
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis('tight')
        ax.axis('off')
        
        if self.latest_function_name == 'get_overview_inventory':
            # 总览库存表格
            table_data = []
            headers = ['货品编码', '质量状态', '总库存', '占用数量', '可用数量']
            
            for item in data:
                available = int(item['qty']) - item['occupied']
                table_data.append([
                    item['skuCode'],
                    item['quality'],
                    str(item['qty']),
                    str(item['occupied']),
                    str(available)
                ])
        else:
            # 物理库存表格
            table_data = []
            headers = ['货品编码', '储位', '库区', '质量', '库存', '占用']
            
            for item in data:
                table_data.append([
                    item['skuCode'],
                    item['binCode'],
                    item['areaName'],
                    item['quality'],
                    str(item['qty']),
                    str(item['occupied'])
                ])
        
        # 创建表格
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # 设置表格样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
            
        fig.suptitle('库存数据表格', fontsize=12, fontweight='bold')
        
        # 嵌入到Tkinter
        canvas = FigureCanvasTkAgg(fig, self.chart_frame_inner)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def show_line_chart(self):
        """显示折线图"""
        if not self.latest_result or not self.latest_result.get('success'):
            messagebox.showwarning("警告", "没有可用的数据生成折线图")
            return
            
        data = self.latest_result.get('data', [])
        if not data:
            return
            
        # 清除之前的图表
        for widget in self.chart_frame_inner.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        if self.latest_function_name == 'get_overview_inventory':
            # 按货品编码显示库存趋势
            sku_codes = [item['skuCode'] for item in data]
            quantities = [item['qty'] for item in data]
            occupied = [item['occupied'] for item in data]
            
            x = range(len(sku_codes))
            ax.plot(x, quantities, marker='o', label='总库存', linewidth=2)
            ax.plot(x, occupied, marker='s', label='占用数量', linewidth=2)
            
            ax.set_xticks(x)
            ax.set_xticklabels(sku_codes, rotation=45)
            ax.set_ylabel('数量')
            ax.set_title('库存趋势图')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # 物理库存按储位显示
            bin_codes = [item['binCode'] for item in data]
            quantities = [item['qty'] for item in data]
            
            x = range(len(bin_codes))
            ax.plot(x, quantities, marker='o', linewidth=2, color='#2196F3')
            
            ax.set_xticks(x)
            ax.set_xticklabels(bin_codes, rotation=45)
            ax.set_ylabel('库存数量')
            ax.set_title('各储位库存分布')
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # 嵌入到Tkinter
        canvas = FigureCanvasTkAgg(fig, self.chart_frame_inner)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def show_pie_chart(self):
        """显示饼图"""
        if not self.latest_result or not self.latest_result.get('success'):
            messagebox.showwarning("警告", "没有可用的数据生成饼图")
            return
            
        data = self.latest_result.get('data', [])
        if not data:
            return
            
        # 清除之前的图表
        for widget in self.chart_frame_inner.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        if self.latest_function_name == 'get_overview_inventory':
            # 按质量状态分组
            quality_counts = Counter(item['quality'] for item in data)
            labels = list(quality_counts.keys())
            sizes = list(quality_counts.values())
            colors = ['#4CAF50', '#FF9800', '#F44336', '#2196F3']
            
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                  colors=colors[:len(labels)], startangle=90)
            ax.set_title('按质量状态分布')
        else:
            # 按库区分组
            area_counts = Counter(item['areaName'] for item in data)
            labels = list(area_counts.keys())
            sizes = list(area_counts.values())
            colors = ['#4CAF50', '#FF9800', '#F44336', '#2196F3', '#9C27B0']
            
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                  colors=colors[:len(labels)], startangle=90)
            ax.set_title('按库区分布')
        
        # 嵌入到Tkinter
        canvas = FigureCanvasTkAgg(fig, self.chart_frame_inner)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def show_bar_chart(self):
        """显示柱状图"""
        if not self.latest_result or not self.latest_result.get('success'):
            messagebox.showwarning("警告", "没有可用的数据生成柱状图")
            return
            
        data = self.latest_result.get('data', [])
        if not data:
            return
            
        # 清除之前的图表
        for widget in self.chart_frame_inner.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        if self.latest_function_name == 'get_overview_inventory':
            # 总览库存柱状图
            sku_codes = [item['skuCode'] for item in data]
            quantities = [item['qty'] for item in data]
            occupied = [item['occupied'] for item in data]
            available = [int(item['qty']) - item['occupied'] for item in data]
            
            x = np.arange(len(sku_codes))
            width = 0.25
            
            ax.bar(x - width, quantities, width, label='总库存', color='#4CAF50')
            ax.bar(x, occupied, width, label='占用数量', color='#FF9800')
            ax.bar(x + width, available, width, label='可用数量', color='#2196F3')
            
            ax.set_xlabel('货品编码')
            ax.set_ylabel('数量')
            ax.set_title('库存对比图')
            ax.set_xticks(x)
            ax.set_xticklabels(sku_codes, rotation=45)
            ax.legend()
        else:
            # 物理库存柱状图
            bin_codes = [item['binCode'] for item in data]
            quantities = [item['qty'] for item in data]
            
            ax.bar(bin_codes, quantities, color='#2196F3')
            ax.set_xlabel('储位编码')
            ax.set_ylabel('库存数量')
            ax.set_title('各储位库存量')
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        # 嵌入到Tkinter
        canvas = FigureCanvasTkAgg(fig, self.chart_frame_inner)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ... 保持所有其他原有方法不变 ...
    
    def add_message(self, role, content, formatted=False):
        """添加消息到聊天显示区域"""
        self.chat_display.config(state=tk.NORMAL)
        
        # 添加时间戳
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if role == "user":
            self.chat_display.insert(tk.END, f"[{timestamp}] 👤 您: ", "user")
            self.chat_display.insert(tk.END, f"{content}\n\n")
        elif role == "assistant":
            self.chat_display.insert(tk.END, f"[{timestamp}] 🤖 助手: ", "assistant")
            self.chat_display.insert(tk.END, f"{content}\n\n")
        elif role == "system":
            self.chat_display.insert(tk.END, f"[{timestamp}] 🔧 系统: ", "system")
            self.chat_display.insert(tk.END, f"{content}\n\n")
        elif role == "error":
            self.chat_display.insert(tk.END, f"[{timestamp}] ❌ 错误: ", "error")
            self.chat_display.insert(tk.END, f"{content}\n\n")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
    def init_clients(self):
        """初始化客户端"""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showerror("错误", "请输入API Key")
            return
            
        try:
            self.llm_client = DashscopeClient(api_key=api_key)
            self.inventory_api = InventoryAPI()
            self.status_var.set("✅ 已初始化")
            self.add_message("system", "客户端初始化成功！现在可以开始对话了。")
        except Exception as e:
            self.status_var.set("❌ 初始化失败")
            self.add_message("error", f"初始化失败: {str(e)}")
            
    def send_message_event(self, event):
        """处理回车键发送消息"""
        self.send_message()
        
    def send_message(self):
        """发送消息"""
        if not self.llm_client:
            messagebox.showwarning("警告", "请先初始化客户端")
            return
            
        user_input = self.input_var.get().strip()
        if not user_input:
            return
            
        # 清空输入框
        self.input_var.set("")
        
        # 显示用户消息
        self.add_message("user", user_input)
        
        # 禁用发送按钮
        self.send_button.config(state=tk.DISABLED)
        self.input_entry.config(state=tk.DISABLED)
        
        # 在新线程中处理消息
        threading.Thread(target=self.process_message, args=(user_input,), daemon=True).start()
        
    def send_quick_query(self, query):
        """发送快捷查询"""
        if not self.llm_client:
            messagebox.showwarning("警告", "请先初始化客户端")
            return
            
        self.input_var.set(query)
        self.send_message()
        
    def process_message(self, user_input):
        """处理消息（在后台线程中运行）"""
        try:
            # 添加用户消息到历史
            self.messages.append({"role": "user", "content": user_input})
            
            # 显示思考状态
            self.root.after(0, lambda: self.add_message("system", "正在思考..."))
            
            # 获取LLM响应
            response = self.llm_client.get_response(self.messages)
            
            if response is None:
                self.root.after(0, lambda: self.add_message("error", "获取响应失败"))
                return
                
            # 检查是否需要处理function call
            if self.llm_client.should_process_function_call(response):
                # 处理function call
                function_call_obj = self.llm_client.fetch_function_call_obj(response)
                
                self.root.after(0, lambda: self.add_message("system", 
                    f"正在调用函数: {function_call_obj.get('name')}"))
                
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
                self.root.after(0, lambda: self.add_message("system", "正在整理结果..."))
                final_response = self.llm_client.get_response(self.messages)
                if final_response:
                    content = self.llm_client.fetch_content(final_response)
                    if content:
                        self.messages.append({"role": "assistant", "content": content})
                        self.root.after(0, lambda: self.add_message("assistant", content))
                        
                        # 显示格式化的结果
                        formatted_result = self.format_inventory_result(
                            function_result, 
                            function_call_obj.get('name')
                        )
                        self.root.after(0, lambda: self.add_message("assistant", formatted_result))
            else:
                # 普通响应
                content = self.llm_client.fetch_content(response)
                if content:
                    self.messages.append({"role": "assistant", "content": content})
                    self.root.after(0, lambda: self.add_message("assistant", content))
                    
        except Exception as e:
            self.root.after(0, lambda: self.add_message("error", f"处理消息时出错: {str(e)}"))
        finally:
            # 重新启用发送按钮
            self.root.after(0, self.enable_input)
            
    def enable_input(self):
        """重新启用输入控件"""
        self.send_button.config(state=tk.NORMAL)
        self.input_entry.config(state=tk.NORMAL)
        self.input_entry.focus()
        
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
        
    def clear_history(self):
        """清空对话历史"""
        self.messages = []
        self.init_system_message()
        
        # 清空聊天显示区域
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # 清空图表区域
        for widget in self.chart_frame_inner.winfo_children():
            widget.destroy()
            
        self.latest_result = None
        self.latest_function_name = None
        
        self.add_message("system", "对话历史已清空")
        
    def show_help(self):
        """显示帮助信息"""
        help_text = """
📋 智能库存助手 - 帮助信息
================================

🔧 功能说明:
• 支持总览库存查询
• 支持物理库存查询
• 支持自然语言对话
• 自动生成数据可视化图表

💡 示例问题:
• "查询苹果的库存情况"
• "冷藏区有什么水果？"
• "查看合格品的总库存"
• "L-01-05储位有什么？"

📊 图表功能:
• 📊 数据表格 - 清晰展示库存数据
• 📈 折线图 - 显示库存趋势
• 🥧 饼图 - 按分类统计占比
• 📊 柱状图 - 对比不同项目数据

🚀 快捷操作:
• 使用左侧快捷查询按钮
• 回车键发送消息
• 点击图表按钮生成可视化
• 清空对话重新开始

⚙️ 使用步骤:
1. 输入Dashscope API Key
2. 点击初始化客户端
3. 开始对话查询库存
4. 查看自动生成的图表

================================
"""
        messagebox.showinfo("帮助", help_text)
        
    def run(self):
        """运行GUI应用"""
        # 设置窗口居中
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # 设置焦点到输入框
        self.input_entry.focus()
        
        # 启动主循环
        self.root.mainloop()

if __name__ == "__main__":
    app = StockChatGUI()
    app.run()