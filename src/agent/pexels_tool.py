#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pexels 图片搜索工具
"""

import os
import requests
import json
from typing import Optional
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger
from model.factory import load_config


def get_pexels_config():
    """获取 Pexels 配置（通过统一的 load_config，支持环境变量覆盖）"""
    return load_config().get('pexels', {})


@tool(description="""从Pexels搜索高质量免费图片。
参数说明：
- query (必须): 搜索关键词，支持中英文（如：sunset, 日落, nature, 自然风景）
- count (可选): 返回图片数量，默认10张，最多15张
- page (可选): 页码，默认为 1

功能说明：
1. 搜索Pexels平台的高质量免费图片
2. 返回图片的URL、尺寸、摄影师信息等
3. 所有图片均为免费商用，无需授权

使用场景：
- 用户想要查找特定主题的图片素材
- 用户需要高质量的免费图片
- 用户想要批量获取某类图片

重要提示：
1. 搜索关键词尽量具体（如："sunset beach" 比 "nature" 更精确）
2. 支持中英文关键词，但英文关键词通常结果更丰富
3. 返回的图片可以直接展示给用户或用于上传
4. 图片来源于Pexels，需要注明摄影师信息

返回格式：
返回JSON格式的图片列表，包含：
- id: 图片ID
- url: 原图URL
- photographer: 摄影师名称
- photographer_url: 摄影师主页
- width: 图片宽度
- height: 图片高度
- avg_color: 平均颜色
""")
def search_pexels_images(query: str, count: Optional[int] = 10, page: Optional[int] = 1) -> str:
    """
    从Pexels搜索图片
    
    Args:
        query: 搜索关键词
        count: 返回图片数量，默认10，最多15
        page: 页码，默认1
    
    Returns:
        JSON格式的搜索结果
    """
    import json
    
    # 加载配置
    config = get_pexels_config()
    api_key = config.get('api_key', '')
    base_url = config.get('base_url', 'https://api.pexels.com/v1')
    max_per_page = config.get('max_per_page', 15)
    
    if not api_key:
        error_msg = "Pexels API密钥未配置，请在 .env 文件中设置 PEXELS_API_KEY 环境变量"
        knowledge_logger.error(f'[PEXELS_SEARCH_ERROR] {error_msg}')
        return json.dumps({
            "error": error_msg,
            "help": "请访问 https://www.pexels.com/api/ 获取免费API密钥，然后在 .env 中配置 PEXELS_API_KEY"
        }, ensure_ascii=False)
    
    # 限制数量
    count = min(count, max_per_page) if count else 10
    
    try:
        knowledge_logger.info(f'[PEXELS_SEARCH] 开始搜索 | 关键词: {query} | 数量: {count}')
        
        # 调用Pexels API
        url = f"{base_url}/search"
        headers = {
            "Authorization": api_key
        }
        params = {
            "query": query,
            "per_page": count,
            "page": page
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            error_msg = f"Pexels API调用失败: HTTP {response.status_code}"
            knowledge_logger.error(f'[PEXELS_SEARCH_ERROR] {error_msg} | Response: {response.text[:200]}')
            return json.dumps({
                "error": error_msg,
                "status_code": response.status_code
            }, ensure_ascii=False)
        
        data = response.json()
        photos = data.get('photos', [])
        
        if not photos:
            knowledge_logger.info(f'[PEXELS_SEARCH] 未找到结果 | 关键词: {query}')
            return json.dumps({
                "total": 0,
                "photos": [],
                "message": f"未找到关于 '{query}' 的图片，请尝试其他关键词"
            }, ensure_ascii=False)
        
        # 提取关键信息（优化：使用中等尺寸作为默认URL以提升聊天性能）
        results = []
        for photo in photos:
            results.append({
                "id": photo.get('id'),
                "url": photo.get('src', {}).get('large'),  # 使用large（中等尺寸）而非original，减少加载时间
                "original_url": photo.get('src', {}).get('original'),  # 保留原图链接供需要时使用
                "large_url": photo.get('src', {}).get('large2x'),
                "medium_url": photo.get('src', {}).get('large'),
                "small_url": photo.get('src', {}).get('medium'),
                "thumbnail_url": photo.get('src', {}).get('small'),
                "photographer": photo.get('photographer'),
                "photographer_url": photo.get('photographer_url'),
                "width": photo.get('width'),
                "height": photo.get('height'),
                "avg_color": photo.get('avg_color'),
                "alt": photo.get('alt', query)
            })
        
        knowledge_logger.info(f'[PEXELS_SEARCH] 搜索成功 | 关键词: {query} | 找到: {len(results)} 张图片')
        
        return json.dumps({
            "total": data.get('total_results', len(results)),
            "count": len(results),
            "query": query,
            "photos": results,
            "message": f"找到 {len(results)} 张关于 '{query}' 的高质量图片"
        }, ensure_ascii=False)
        
    except requests.exceptions.Timeout:
        error_msg = "Pexels API请求超时，请稍后重试"
        knowledge_logger.error(f'[PEXELS_SEARCH_ERROR] {error_msg}')
        return json.dumps({"error": error_msg}, ensure_ascii=False)
    except Exception as e:
        error_msg = f"搜索图片时发生错误: {str(e)}"
        knowledge_logger.error(f'[PEXELS_SEARCH_ERROR] {error_msg}')
        import traceback
        knowledge_logger.error(f'[PEXELS_SEARCH_ERROR] 堆栈: {traceback.format_exc()}')
        return json.dumps({"error": error_msg}, ensure_ascii=False)


@tool(description="""获取Pexels精选图片（Curated Photos）。
参数说明：
- count (可选): 返回图片数量，默认10张，最多15张
- page (可选): 页码，默认为 1

功能说明：
1. 获取Pexels编辑精选的高质量图片
2. 这些图片经过人工筛选，质量有保证
3. 适合用于获取灵感或随机浏览

使用场景：
- 用户想要浏览高质量图片
- 用户没有明确的搜索目标
- 用户想要获取设计灵感

返回格式：
与search_pexels_images相同的JSON格式
""")
def get_pexels_curated(count: Optional[int] = 10, page: Optional[int] = 1) -> str:
    """
    获取Pexels精选图片
    
    Args:
        count: 返回图片数量，默认10，最多15
        page: 页码，默认1
    
    Returns:
        JSON格式的图片列表
    """
    import json
    
    # 加载配置
    config = get_pexels_config()
    api_key = config.get('api_key', '')
    base_url = config.get('base_url', 'https://api.pexels.com/v1')
    max_per_page = config.get('max_per_page', 15)
    
    if not api_key:
        error_msg = "Pexels API密钥未配置"
        knowledge_logger.error(f'[PEXELS_CURATED_ERROR] {error_msg}')
        return json.dumps({"error": error_msg}, ensure_ascii=False)
    
    # 限制数量
    count = min(count, max_per_page) if count else 10
    
    try:
        knowledge_logger.info(f'[PEXELS_CURATED] 获取精选图片 | 数量: {count}')
        
        # 调用Pexels API
        url = f"{base_url}/curated"
        headers = {
            "Authorization": api_key
        }
        params = {
            "per_page": count,
            "page": page
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            error_msg = f"Pexels API调用失败: HTTP {response.status_code}"
            knowledge_logger.error(f'[PEXELS_CURATED_ERROR] {error_msg}')
            return json.dumps({"error": error_msg}, ensure_ascii=False)
        
        data = response.json()
        photos = data.get('photos', [])
        
        # 提取关键信息（优化：使用中等尺寸作为默认URL以提升聊天性能）
        results = []
        for photo in photos:
            results.append({
                "id": photo.get('id'),
                "url": photo.get('src', {}).get('large'),  # 使用large（中等尺寸）而非original，减少加载时间
                "original_url": photo.get('src', {}).get('original'),  # 保留原图链接供需要时使用
                "large_url": photo.get('src', {}).get('large2x'),
                "medium_url": photo.get('src', {}).get('large'),
                "small_url": photo.get('src', {}).get('medium'),
                "thumbnail_url": photo.get('src', {}).get('small'),
                "photographer": photo.get('photographer'),
                "photographer_url": photo.get('photographer_url'),
                "width": photo.get('width'),
                "height": photo.get('height'),
                "avg_color": photo.get('avg_color'),
                "alt": photo.get('alt', 'Curated photo')
            })
        
        knowledge_logger.info(f'[PEXELS_CURATED] 获取成功 | 数量: {len(results)}')
        
        return json.dumps({
            "count": len(results),
            "photos": results,
            "message": f"获取了 {len(results)} 张精选图片"
        }, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"获取精选图片时发生错误: {str(e)}"
        knowledge_logger.error(f'[PEXELS_CURATED_ERROR] {error_msg}')
        return json.dumps({"error": error_msg}, ensure_ascii=False)
