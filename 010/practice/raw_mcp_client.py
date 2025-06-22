#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原生MCP客户端实现
不使用任何agent框架，直接与MCP服务器进行JSON-RPC 2.0通信
展示完整的MCP交互流程
"""

import json
import subprocess
import asyncio
import os
import sys
from typing import Dict, Any, Optional, List
import uuid

class MCPClient:
    def __init__(self):
        self.process = None
        self.request_id = 0
        self.tools = []
        
    def get_next_id(self) -> int:
        """获取下一个请求ID"""
        self.request_id += 1
        return self.request_id
    
    async def start_mcp_server(self, command: str, args: List[str], env: Dict[str, str] = None):
        """启动MCP服务器进程"""
        print(f"🚀 启动MCP服务器: {command} {' '.join(args)}")
        
        # 设置环境变量
        server_env = os.environ.copy()
        if env:
            server_env.update(env)
            
        # 启动子进程
        self.process = await asyncio.create_subprocess_exec(
            command, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=server_env
        )
        
        print(f"✅ MCP服务器已启动，PID: {self.process.pid}")
        
    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """发送JSON-RPC 2.0请求"""
        request_id = self.get_next_id()
        
        # 构建JSON-RPC 2.0请求
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        
        if params is not None:
            request["params"] = params
            
        # 序列化请求
        request_json = json.dumps(request, ensure_ascii=False)
        print(f"\n📤 发送请求 (ID: {request_id}):")
        print(f"   方法: {method}")
        print(f"   参数: {json.dumps(params, ensure_ascii=False, indent=2) if params else 'None'}")
        print(f"   原始JSON: {request_json}")
        
        # 发送请求（添加换行符）
        self.process.stdin.write((request_json + "\n").encode('utf-8'))
        await self.process.stdin.drain()
        
        # 读取响应
        response_line = await self.process.stdout.readline()
        response_json = response_line.decode('utf-8').strip()
        
        print(f"\n📥 收到响应 (ID: {request_id}):")
        print(f"   原始JSON: {response_json}")
        
        try:
            response = json.loads(response_json)
            print(f"   解析结果: {json.dumps(response, ensure_ascii=False, indent=2)}")
            return response
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"   响应内容: {response_json}")
            raise
    
    async def initialize(self) -> Dict[str, Any]:
        """初始化MCP连接"""
        print("\n🔄 步骤1: 初始化MCP连接")
        
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "raw-mcp-client",
                "version": "1.0.0"
            }
        }
        
        response = await self.send_request("initialize", params)
        
        if "error" in response:
            raise Exception(f"初始化失败: {response['error']}")
            
        print(f"✅ 初始化成功")
        print(f"   服务器信息: {response['result'].get('serverInfo', {})}")
        print(f"   协议版本: {response['result'].get('protocolVersion')}")
        print(f"   服务器能力: {response['result'].get('capabilities', {})}")
        
        return response['result']
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """获取工具列表"""
        print("\n🔄 步骤2: 获取工具列表")
        
        response = await self.send_request("tools/list")
        
        if "error" in response:
            raise Exception(f"获取工具列表失败: {response['error']}")
            
        self.tools = response['result']['tools']
        
        print(f"✅ 获取到 {len(self.tools)} 个工具:")
        for i, tool in enumerate(self.tools, 1):
            print(f"   {i}. {tool['name']}: {tool['description']}")
            if 'inputSchema' in tool:
                required = tool['inputSchema'].get('required', [])
                properties = tool['inputSchema'].get('properties', {})
                print(f"      必需参数: {required}")
                print(f"      所有参数: {list(properties.keys())}")
        
        return self.tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        print(f"\n🔄 步骤3: 调用工具 '{tool_name}'")
        
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        
        response = await self.send_request("tools/call", params)
        
        if "error" in response:
            raise Exception(f"工具调用失败: {response['error']}")
            
        result = response['result']
        print(f"✅ 工具调用成功")
        print(f"   是否出错: {result.get('isError', False)}")
        
        if 'content' in result:
            print(f"   返回内容:")
            for content in result['content']:
                if content['type'] == 'text':
                    print(f"     文本: {content['text'][:200]}{'...' if len(content['text']) > 200 else ''}")
                else:
                    print(f"     {content['type']}: {content}")
        
        return result
    
    async def close(self):
        """关闭MCP连接"""
        if self.process:
            print("\n🔄 关闭MCP服务器")
            self.process.stdin.close()
            await self.process.wait()
            print("✅ MCP服务器已关闭")

async def demo_amap_interaction():
    """演示与高德地图MCP服务器的完整交互"""
    print("=" * 60)
    print("🗺️  高德地图MCP服务器交互演示")
    print("=" * 60)
    
    # 检查环境变量
    amap_api_key = os.getenv('AMAP_API_KEY')
    if not amap_api_key:
        print("❌ 错误: 请设置 AMAP_API_KEY 环境变量")
        return
    
    client = MCPClient()
    
    try:
        # 启动MCP服务器
        await client.start_mcp_server(
            "npx",
            ["-y", "@amap/amap-maps-mcp-server"],
            {"AMAP_MAPS_API_KEY": amap_api_key}
        )
        
        # 等待服务器启动
        await asyncio.sleep(2)
        
        # 初始化连接
        server_info = await client.initialize()
        
        # 获取工具列表
        tools = await client.list_tools()
        
        # 演示工具调用
        print("\n" + "=" * 60)
        print("🎯 开始演示工具调用")
        print("=" * 60)
        
        # 示例1: POI搜索
        if any(tool['name'] == 'poi_search' for tool in tools):
            print("\n🔍 示例1: 搜索东方明珠")
            result1 = await client.call_tool('poi_search', {
                'keywords': '东方明珠',
                'city': '上海'
            })
        
        # 示例2: 地理编码
        if any(tool['name'] == 'geocoding' for tool in tools):
            print("\n📍 示例2: 地理编码查询")
            result2 = await client.call_tool('geocoding', {
                'address': '上海市浦东新区世纪大道1号',
                'city': '上海'
            })
        
        # 示例3: 天气查询
        if any(tool['name'] == 'weather_info' for tool in tools):
            print("\n🌤️ 示例3: 天气查询")
            result3 = await client.call_tool('weather_info', {
                'city': '上海'
            })
        
        print("\n" + "=" * 60)
        print("✅ 所有演示完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.close()

async def interactive_mode():
    """交互模式"""
    print("=" * 60)
    print("🎮 MCP交互模式")
    print("=" * 60)
    
    # 检查环境变量
    amap_api_key = os.getenv('AMAP_API_KEY')
    if not amap_api_key:
        print("❌ 错误: 请设置 AMAP_API_KEY 环境变量")
        return
    
    client = MCPClient()
    
    try:
        # 启动MCP服务器
        # 在 interactive_mode 函数中，将原来的启动方式：
        # await client.start_mcp_server(
        #     "npx",
        #     ["-y", "@amap/amap-maps-mcp-server"],
        #     {"AMAP_MAPS_API_KEY": amap_api_key}
        # )
        
        # 修改为：
        await client.start_mcp_server(
            "cmd",
            ["/c", "npx", "-y", "@amap/amap-maps-mcp-server"],
            {"AMAP_MAPS_API_KEY": amap_api_key}
        )
        
        await asyncio.sleep(2)
        await client.initialize()
        tools = await client.list_tools()
        
        print("\n可用命令:")
        print("  list - 显示所有工具")
        print("  call <tool_name> <json_args> - 调用工具")
        print("  exit - 退出")
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command == "exit":
                    break
                elif command == "list":
                    print("\n可用工具:")
                    for i, tool in enumerate(tools, 1):
                        print(f"  {i}. {tool['name']}: {tool['description']}")
                elif command.startswith("call "):
                    parts = command.split(" ", 2)
                    if len(parts) >= 3:
                        tool_name = parts[1]
                        try:
                            args = json.loads(parts[2])
                            await client.call_tool(tool_name, args)
                        except json.JSONDecodeError:
                            print("❌ 参数必须是有效的JSON格式")
                    else:
                        print("❌ 用法: call <tool_name> <json_args>")
                else:
                    print("❌ 未知命令")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 自动演示模式")
    print("2. 交互模式")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        asyncio.run(demo_amap_interaction())
    elif choice == "2":
        asyncio.run(interactive_mode())
    else:
        print("无效选择")