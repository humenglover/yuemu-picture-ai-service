"""
实时天气及未来气象预报查询工具
基于 wttr.in 免费接口，无需 API Key，全球城市通用
"""
import requests
from typing import Optional
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger


@tool
def weather_query_tool(location: str) -> str:
    """
    全能天气气象查询工具 - 查询指定城市的当前实时天气以及未来3天预报
    
    适用场景：
    - 用户询问“今天天气怎么样”、“未来几天天气”、“紫外线强度”、“出门要带伞吗”、“日出日落时间”等问题时调用。
    - 支持中英文城市名（如：北京、Beijing、西安、Shanghai）。
    
    Args:
        location: 城市名称（必填），支持中英文，例如：西安、北京、Shanghai、New York
    
    Returns:
        str: 丰富的结构化天气报告，包含：实时天气状况、温湿度、紫外线，以及未来3天的最高/最低气温、天气描述、日出日落时间等。
    
    示例：
        - weather_query_tool("北京")
    """
    try:
        # 构建请求 URL，format=j1 返回 JSON 格式
        url = f"https://wttr.in/{location}?format=j1"
        
        # 设置请求头，模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        
        knowledge_logger.info(f"[WEATHER_TOOL] 正在查询城市 [{location}] 的全景天气信息...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 解析 JSON 数据
        data = response.json()
        
        # --- 1. 提取实时天气 ---
        current_condition = data.get('current_condition', [{}])[0]
        weather_desc = current_condition.get('weatherDesc', [{}])[0].get('value', '未知')
        temp_c = current_condition.get('temp_C', '未知')
        feels_like_c = current_condition.get('FeelsLikeC', '未知')
        humidity = current_condition.get('humidity', '未知')
        wind_speed_kmph = current_condition.get('windspeedKmph', '未知')
        wind_dir = current_condition.get('winddir16Point', '未知')
        uv_index_current = current_condition.get('uvIndex', '未知')
        
        result_parts = [
            f"【{location}】气象报告：",
            f"--- 当前实时气象数据 ---",
            f"天气状况：{weather_desc}",
            f"实时温度：{temp_c}°C (体感: {feels_like_c}°C)",
            f"相对湿度：{humidity}% | 风向风速：{wind_dir} {wind_speed_kmph} km/h",
            f"紫外线指数：{uv_index_current}",
            ""
        ]
        
        # --- 2. 提取未来 3 天预报 ---
        weather_forecasts = data.get('weather', [])
        if weather_forecasts:
            result_parts.append("--- 未来 3 日气象预报 ---")
            for forecast in weather_forecasts:
                date = forecast.get('date', '未知日期')
                max_t = forecast.get('maxtempC', '?')
                min_t = forecast.get('mintempC', '?')
                uv = forecast.get('uvIndex', '?')
                
                # 提取日出日落
                astronomy = forecast.get('astronomy', [{}])[0]
                sunrise = astronomy.get('sunrise', '?')
                sunset = astronomy.get('sunset', '?')
                
                # 提取当天的整体天气概况（取中午时段的描述作为代表）
                hourly = forecast.get('hourly', [])
                if hourly:
                    mid_day = hourly[len(hourly)//2]
                    day_desc = mid_day.get('weatherDesc', [{}])[0].get('value', '未知')
                else:
                    day_desc = '未知'
                
                result_parts.append(
                    f"- {date}: {day_desc}，气温 {min_t}°C ~ {max_t}°C，"
                    f"紫外线指数 {uv}，日出 {sunrise}，日落 {sunset}"
                )
                
        knowledge_logger.info(f"[WEATHER_TOOL] 成功获取 [{location}] 综合天气信息")
        return "\n".join(result_parts)
        
    except requests.exceptions.Timeout:
        error_msg = f"查询超时：无法连接到天气服务，请稍后重试"
        knowledge_logger.error(f"[WEATHER_TOOL] 天气查询超时 - 城市: {location}")
        return error_msg
        
    except requests.exceptions.RequestException as e:
        error_msg = f"网络错误：{str(e)}"
        knowledge_logger.error(f"[WEATHER_TOOL] 天气查询失败 - 城市: {location}, 错误: {str(e)}")
        return error_msg
        
    except (KeyError, IndexError, ValueError) as e:
        error_msg = f"数据解析失败：可能城市名称不正确或服务异常"
        knowledge_logger.error(f"[WEATHER_TOOL] 天气数据解析失败 - 城市: {location}, 错误: {str(e)}")
        return error_msg
        
    except Exception as e:
        error_msg = f"未知错误：{str(e)}"
        knowledge_logger.error(f"[WEATHER_TOOL] 天气查询异常 - 城市: {location}, 错误: {str(e)}")
        return error_msg
