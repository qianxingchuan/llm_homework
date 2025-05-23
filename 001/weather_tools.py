from api_key_constants import weather_secret
import requests
import json


# header添加X-QW-Api-Key
headers = {
    "X-QW-Api-Key": weather_secret.get("api_key")
}

# add new function: today
def today():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d")

# 新增方法：通过IP获取当前地址
def get_location_by_ip():
    try:
        response = requests.get('https://my.ip.cn/json/')
        response.raise_for_status()
        data = response.json()
        city = data.get('data').get('city',None)
        if city is None:
            return None
        return get_location_id_by_city_name(city)
    except requests.RequestException as e:
        print(f"请求出错: {str(e)}")
        return None
    except (KeyError, ValueError):
        print("解析响应数据出错")
        return None


# 新增方法：获取天气
def get_current_weather(locationId,date):
    # 这边校验一下date参数，如果不是今天，抛出异常
    today_str = today();
    if date  != today_str:
        raise Exception("日期必须是今天，今天是"+today_str+",输入是"+date)
    if locationId is None:
        return None
    url="https://m662b5hwcm.re.qweatherapi.com/v7/weather/now?location="+locationId;
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        weatherObj = response.json()
        return weatherObj.get('now',None)
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析出错: {e}")
        return None

# 查询城市的id
def get_location_id_by_city_name(city_name):
    url = "https://m662b5hwcm.re.qweatherapi.com/geo/v2/city/lookup?location="+city_name;
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        city_obj = response.json()
        locations = city_obj.get('location',[])
        if locations:
            return locations[0].get('id', None)
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析出错: {e}")
        return None

