#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能特征裁剪工具 - 基于 OpenCV 边缘密度提取视觉重心 (极低资源消耗)
无需深度学习，自动计算画面主体位置，执行裁剪并上传到服务器
"""
import os
import cv2
import yaml
import json
import requests
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger
from .context import get_sa_token


def load_tool_config() -> dict:
    """加载工具配置文件"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'tool.yml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        knowledge_logger.warning(f'[SMART_CROP] 加载配置文件失败: {str(e)}')
    return {}


def calculate_crop_box(orig_w: int, orig_h: int, cx: int, cy: int, target_ratio: float) -> tuple:
    """
    根据视觉中心和目标比例，计算最大可能的安全裁剪框
    返回: (x1, y1, x2, y2)
    """
    # 1. 根据原图和目标比例，确定裁剪框的绝对大小
    current_ratio = orig_w / orig_h
    
    if current_ratio > target_ratio:
        # 图片太宽，以高度为基准截取宽度
        crop_h = orig_h
        crop_w = int(orig_h * target_ratio)
    else:
        # 图片太高，以宽度为基准截取高度
        crop_w = orig_w
        crop_h = int(orig_w / target_ratio)
    
    # 2. 以视觉中心 (cx, cy) 为准，初步计算左上角坐标
    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    
    # 3. 边界碰撞检测与修正 (防越界)
    if x1 < 0:
        x1 = 0
    elif x1 + crop_w > orig_w:
        x1 = orig_w - crop_w
        
    if y1 < 0:
        y1 = 0
    elif y1 + crop_h > orig_h:
        y1 = orig_h - crop_h
    
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    
    return (x1, y1, x2, y2)


def analyze_visual_center(image_bgr: np.ndarray) -> tuple:
    """
    分析图片的视觉重心
    返回: (cx, cy) 相对于原图的坐标
    """
    orig_h, orig_w = image_bgr.shape[:2]
    
    # 核心优化：极速微缩降维
    # 将图片压缩到最长边 300px 以内进行计算，极大降低 CPU 开销
    max_dim = 300.0
    scale = 1.0
    
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        small_w, small_h = int(orig_w * scale), int(orig_h * scale)
        # 使用最近邻插值，速度最快
        small_img = cv2.resize(image_bgr, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    else:
        small_img = image_bgr
        scale = 1.0
    
    # 视觉重心提取计算 (Canny 边缘 + 图像矩)
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    
    # 提取边缘线条 (信息密集区)
    edges = cv2.Canny(gray, 50, 150)
    
    # 用高斯模糊把细碎的线条连成一片"能量云"
    blurred_edges = cv2.GaussianBlur(edges, (21, 21), 0)
    
    # 计算这团"能量云"的重心 (Center of Mass)
    M = cv2.moments(blurred_edges)
    
    if M["m00"] != 0:
        cx_small = int(M["m10"] / M["m00"])
        cy_small = int(M["m01"] / M["m00"])
    else:
        # 万一是纯白/纯黑的极简图片，找不到边缘，默认取正中心
        cx_small, cy_small = small_img.shape[1] // 2, small_img.shape[0] // 2
    
    # 坐标映射回原图
    cx_orig = int(cx_small / scale)
    cy_orig = int(cy_small / scale)
    
    return (cx_orig, cy_orig)


@tool(description="""智能裁剪图片并上传到服务器。

功能说明：
1. 自动分析图片中信息量最大、细节最丰富的区域（视觉重心）
2. 在保证不裁剪掉核心主体的前提下，执行智能裁剪
3. 将裁剪后的图片上传到服务器，返回新图片的 URL 和元数据

支持的裁剪比例：
- "1:1" - 正方形（适合头像、Instagram）
- "3:4" - 竖版（适合小红书、抖音）
- "4:3" - 横版标准（适合相机照片）
- "16:9" - 宽屏（适合视频封面、横屏壁纸）
- "9:16" - 竖屏（适合手机壁纸、竖屏视频）
- "2:3" - 竖版照片（经典照片比例）
- "3:2" - 横版照片（经典照片比例）

使用场景：
- 用户要求"帮我把这张图裁成头像"或"生成一张合适的横版封面"
- 为图库自动生成不会"切头"的高质量封面图/缩略图
- 需要将图片裁剪成特定比例并保存

参数说明：
- image_url (必需): 原图片的 URL 地址
- aspect_ratio (可选): 目标裁剪比例，支持 "1:1", "3:4", "4:3", "16:9", "9:16", "2:3", "3:2"，默认为 "1:1"
- space_id (可选): 上传到的目标空间ID，None或0表示公共空间，-1表示帖子图片
- name (可选): 裁剪后图片的名称
- introduction (可选): 图片描述

返回字段：
返回裁剪分析报告和上传后的图片信息，包含：url、thumbnailUrl、name、introduction、tags、category、picSize、picWidth、picHeight、picScale、picFormat、picColor、crop_info（裁剪详情）
""")
def smart_crop_and_upload(
    image_url: str,
    aspect_ratio: str = "1:1",
    space_id: Optional[int] = None,
    name: Optional[str] = None,
    introduction: Optional[str] = None
) -> str:
    """智能裁剪图片并上传到服务器"""
    
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)，无法执行上传操作。"}'
    
    try:
        knowledge_logger.info(f'[SMART_CROP] 开始智能裁剪 | URL: {image_url} | 比例: {aspect_ratio}')
        
        if not image_url:
            return "请提供有效的图片 URL。"
        
        # 支持的裁剪比例映射
        ratio_map = {
            "1:1": 1.0,      # 正方形
            "3:4": 0.75,     # 竖版（小红书）
            "4:3": 1.333,    # 横版标准
            "16:9": 1.778,   # 宽屏
            "9:16": 0.5625,  # 竖屏
            "2:3": 0.667,    # 竖版照片
            "3:2": 1.5       # 横版照片
        }
        
        # 验证裁剪比例
        if aspect_ratio not in ratio_map:
            supported = ", ".join(ratio_map.keys())
            return f'{{"error": "不支持的裁剪比例 {aspect_ratio}。支持的比例：{supported}"}}'
        
        target_ratio_val = ratio_map[aspect_ratio]
        
        # 1. 下载原图 (防盗链绕过)
        config = load_tool_config()
        headers = {'User-Agent': 'Mozilla/5.0'}
        yuemu_config = config.get('yuemu', {})
        
        if yuemu_config.get('cdn_domain', 'static.yuemutuku.com') in image_url:
            headers['Referer'] = yuemu_config.get('website_url', 'https://www.yuemutuku.com')
        
        response = requests.get(image_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        pil_image = Image.open(BytesIO(response.content))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 转为 OpenCV BGR 格式
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = image_bgr.shape[:2]
        
        knowledge_logger.info(f'[SMART_CROP] 原图尺寸: {orig_w}x{orig_h}')
        
        # 2. 分析视觉重心
        cx_orig, cy_orig = analyze_visual_center(image_bgr)
        
        # 3. 计算裁剪框
        x1, y1, x2, y2 = calculate_crop_box(orig_w, orig_h, cx_orig, cy_orig, target_ratio_val)
        
        knowledge_logger.info(f'[SMART_CROP] 视觉重心: ({cx_orig}, {cy_orig}) | 裁剪框: [{x1},{y1},{x2},{y2}]')
        
        # 4. 执行裁剪
        cropped_bgr = image_bgr[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped_rgb)
        
        crop_w, crop_h = x2 - x1, y2 - y1
        knowledge_logger.info(f'[SMART_CROP] 裁剪后尺寸: {crop_w}x{crop_h}')
        
        # 5. 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            cropped_pil.save(tmp_path, 'JPEG', quality=95)
            
            # 6. 上传到服务器
            try:
                from model.factory import load_config
                rag_config = load_config()
                java_base_url = rag_config.get("java_backend_url", "http://127.0.0.1:8123/api")
            except Exception:
                java_base_url = "http://127.0.0.1:8123/api"
            
            upload_url = f"{java_base_url}/picture/upload"
            
            # 构建请求
            upload_headers = {"satoken": token}
            
            params = {}
            if space_id is not None:
                params['spaceId'] = space_id
            if name:
                params['picName'] = name
            if introduction:
                params['introduction'] = introduction
                
            # 引入公共工具并追加图片天然宽高及主色调，确保入库信息完整
            from utils.image_utils import analyze_local_image_attributes
            attrs = analyze_local_image_attributes(tmp_path)
            if attrs:
                params['picWidth'] = attrs['picWidth']
                params['picHeight'] = attrs['picHeight']
                params['picScale'] = attrs['picScale']
                params['picColor'] = attrs['picColor']
            
            # 上传文件
            with open(tmp_path, 'rb') as f:
                files = {'file': (f'cropped_{aspect_ratio.replace(":", "x")}.jpg', f, 'image/jpeg')}
                
                knowledge_logger.info(f'[SMART_CROP] 开始上传裁剪后的图片...')
                upload_response = requests.post(
                    upload_url,
                    headers=upload_headers,
                    params=params,
                    files=files,
                    timeout=60
                )
                
                upload_result = upload_response.json()
            
            # 8. 处理上传结果
            if upload_response.status_code == 200 and upload_result.get("code") == 0:
                data = upload_result.get("data", {})
                pic_url = data.get("url")
                thumbnail_url = data.get("thumbnailUrl")
                
                knowledge_logger.info(f'[SMART_CROP] 裁剪并上传成功 | URL: {pic_url}')
                
                # 构建返回报告
                report_parts = ["**智能裁剪完成**\n\n"]
                report_parts.append(f"**原图尺寸**: {orig_w} × {orig_h} px\n")
                
                cx_percent = round((cx_orig / orig_w) * 100)
                cy_percent = round((cy_orig / orig_h) * 100)
                report_parts.append(f"**视觉重心**: 位于画面宽度 {cx_percent}%，高度 {cy_percent}% 处\n")
                report_parts.append(f"**裁剪比例**: {aspect_ratio}\n")
                report_parts.append(f"**裁剪后尺寸**: {crop_w} × {crop_h} px\n\n")
                report_parts.append("**已上传到服务器**\n\n")
                
                # 返回图片信息（JSON格式，供前端解析）
                result_data = {
                    "type": "image_cropped",
                    "msg": "".join(report_parts),
                    "url": pic_url,
                    "thumbnailUrl": thumbnail_url,
                    "name": data.get("name"),
                    "introduction": data.get("introduction"),
                    "tags": data.get("tags", []),
                    "category": data.get("category"),
                    "picSize": data.get("picSize"),
                    "picWidth": data.get("picWidth"),
                    "picHeight": data.get("picHeight"),
                    "picScale": data.get("picScale"),
                    "picFormat": data.get("picFormat"),
                    "picColor": data.get("picColor"),
                    "crop_info": {
                        "original_size": f"{orig_w}x{orig_h}",
                        "cropped_size": f"{crop_w}x{crop_h}",
                        "visual_center": f"({cx_orig}, {cy_orig})",
                        "crop_box": [x1, y1, x2, y2],
                        "aspect_ratio": aspect_ratio
                    }
                }
                
                return json.dumps(result_data, ensure_ascii=False)
            else:
                msg = upload_result.get("message", "未知错误")
                knowledge_logger.error(f'[SMART_CROP] 上传失败: {msg}')
                return f'{{"error": "上传裁剪后的图片失败: {msg}"}}'
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    except Exception as e:
        error_msg = f"智能裁剪处理失败: {str(e)}"
        knowledge_logger.error(f'[SMART_CROP] {error_msg}')
        return f'{{"error": "{error_msg}"}}'
