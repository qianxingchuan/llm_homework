import asyncio
import json
import subprocess
from aiohttp import web, ClientSession
import aiohttp_cors

class MCPHTTPProxy:
    def __init__(self, mcp_command, mcp_args, port=8080):
        self.mcp_command = mcp_command
        self.mcp_args = mcp_args
        self.port = port
        self.mcp_process = None
        
    async def start_mcp_server(self):
        """启动 MCP 服务器进程"""
        self.mcp_process = await asyncio.create_subprocess_exec(
            self.mcp_command, *self.mcp_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
    async def send_mcp_request(self, request_data):
        """向 MCP 服务器发送请求"""
        if not self.mcp_process:
            await self.start_mcp_server()
            
        # 发送请求到 MCP 服务器
        request_json = json.dumps(request_data) + '\n'
        self.mcp_process.stdin.write(request_json.encode())
        await self.mcp_process.stdin.drain()
        
        # 读取响应
        response_line = await self.mcp_process.stdout.readline()
        response_data = json.loads(response_line.decode().strip())
        return response_data
        
    async def handle_request(self, request):
        """处理 HTTP 请求"""
        try:
            request_data = await request.json()
            response_data = await self.send_mcp_request(request_data)
            return web.json_response(response_data)
        except Exception as e:
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
            
    async def create_app(self):
        """创建 web 应用"""
        app = web.Application()
        
        # 设置 CORS
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # 添加路由
        route = app.router.add_post('/', self.handle_request)
        cors.add(route)
        
        return app
        
    async def run(self):
        """运行代理服务器"""
        app = await self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        print(f"MCP HTTP 代理服务器启动在 http://localhost:{self.port}")
        
        try:
            await asyncio.Future()  # 保持运行
        except KeyboardInterrupt:
            print("正在关闭服务器...")
        finally:
            if self.mcp_process:
                self.mcp_process.terminate()
                await self.mcp_process.wait()

if __name__ == '__main__':
    # 配置 MCP 服务器
    proxy = MCPHTTPProxy(
        mcp_command='python',
        mcp_args=['e:\\code\\llm_homework\\010\\CASE-MCP Demo-1\\txt_counter.py'],
        port=8080
    )
    
    asyncio.run(proxy.run())