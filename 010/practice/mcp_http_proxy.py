from fastmcp import FastMCP, Client
import sys
import os

# 配置后端 MCP 服务器（您的 txt_counter.py）
backend_client = Client("e:\\code\\llm_homework\\010\\CASE-MCP Demo-1\\txt_counter.py")

# 创建代理服务器
proxy_server = FastMCP.from_client(
    backend_client,
    name="TxtCounterProxy"
)

if __name__ == "__main__":
    # 以 HTTP 模式运行代理服务器
    proxy_server.run(transport="sse", port=8080)