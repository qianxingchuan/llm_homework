from fastapi import FastAPI, HTTPException
from datetime import date
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Agent Card声明（通过/.well-known/agent.json暴露）
WEATHER_AGENT_CARD = {
    "name": "WeatherAgent",
    "version": "1.0",
    "description": "提供指定日期的天气数据查询",
    "endpoints": {
        "task_submit": "/api/tasks/weather",
        "sse_subscribe": "/api/tasks/updates"
    },
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "format": "date"},
            "location": {"type": "string", "enum": ["北京"]}
        },
        "required": ["date"]
    },
    "authentication": {"methods": ["API_Key"]}
}

# 任务请求模型
class WeatherTaskRequest(BaseModel):
    task_id: str
    params: dict

# 模拟天气数据存储
weather_db = {
    "2025-05-08": {"temperature": "25℃", "condition": "雷阵雨"},
    "2025-05-09": {"temperature": "18℃", "condition": "小雨转晴"},
    "2025-05-10": {"temperature": "22℃", "condition": "多云转晴"}
}

@app.get("/.well-known/agent.json")
async def get_agent_card():
    return WEATHER_AGENT_CARD

@app.post("/api/tasks/weather")
async def handle_weather_task(request: WeatherTaskRequest):
    """处理天气查询任务"""
    target_date = request.params.get("date")
    # 参数验证
    if not target_date or target_date not in weather_db:
        raise HTTPException(status_code=400, detail="无效日期参数")
    return {
        "task_id": request.task_id,
        "status": "completed",
        "artifact": {
            "date": target_date,
            "weather": weather_db[target_date]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)