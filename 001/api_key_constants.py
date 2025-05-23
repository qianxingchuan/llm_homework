import os
# 保存各种apikey
dashscope_secret = {
    "api_key" : os.getenv("DASHSCOPE_APP_KEY")
}

weather_secret = {
    "api_key" : os.getenv("WEATHER_APP_KEY")
}