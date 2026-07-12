#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键画质增强工具 (Pro-Enhance)
使用高级传统图像处理算法：自适应 Gamma 校正 + CLAHE 局部提亮 + USM 专业锐化 + 非局部均值降噪
提升图片对比度、细节和色彩饱和度，无需大模型也能达到极佳的通透感。
"""
import os
import cv2
import math
import yaml
import json
import requests
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance
from typing import Optional
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .context import get_sa_token

def load_tool_config() -> dict:
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'tool.yml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        knowledge_logger.warning(f'[IMAGE_ENHANCE] 加载配置文件失败: {str(e)}')
    return {}

def _get_requests_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def enhance_image_quality(image_bgr: np.ndarray, enhance_level: str = "medium") -> np.ndarray:
    """
    高级画质增强流水线
    """
    height, width = image_bgr.shape[:2]
    
    # 0. 如果是 strong 级别，加入降噪处理 (非局部均值滤波)
    if enhance_level == "strong" and width * height < 3000 * 3000: # 防内存爆炸
        # 参数 h 越小细节保留越多
        image_bgr = cv2.fastNlMeansDenoisingColored(image_bgr, None, 3, 3, 7, 21)

    # 1. 自适应 Gamma 校正 (修正过曝或欠曝)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_illumination = np.mean(gray)
    if mean_illumination > 0:
        # 将平均亮度映射到理想的 128 (中灰度)
        gamma = math.log(128 / 255.0) / math.log(mean_illumination / 255.0)
        # 限制 Gamma 防止画面过度失真
        gamma = np.clip(gamma, 0.6, 1.4)
        if abs(gamma - 1.0) > 0.05:
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            image_bgr = cv2.LUT(image_bgr, table)

    # 2. LAB 空间 CLAHE 增强 (保留色彩的前提下提升局部细节)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clip_limit = {"light": 1.5, "medium": 2.5, "strong": 3.5}.get(enhance_level, 2.5)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    lab_enhanced = cv2.merge((l_channel, a_channel, b_channel))
    image_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. 色彩饱和度与全局对比度提升 (Pillow)
    rgb_enhanced = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_enhanced)
    
    color_factor = {"light": 1.15, "medium": 1.3, "strong": 1.45}.get(enhance_level, 1.3)
    contrast_factor = {"light": 1.05, "medium": 1.15, "strong": 1.25}.get(enhance_level, 1.15)
    
    pil_image = ImageEnhance.Color(pil_image).enhance(color_factor)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast_factor)
    
    # 4. USM 专业锐化 (Unsharp Masking)
    # 原理：原图叠加一层(原图-模糊图)的细节差异
    final_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gaussian = cv2.GaussianBlur(final_bgr, (0, 0), 2.0)
    
    amount = {"light": 1.2, "medium": 1.4, "strong": 1.7}.get(enhance_level, 1.4)
    # result = original * amount + blurred * (1 - amount)
    final_bgr = cv2.addWeighted(final_bgr, amount, gaussian, 1.0 - amount, 0)
    
    return final_bgr

@tool(description="""一键画质增强工具（Pro 版），提升图片的对比度、色彩饱和度、光影和清晰度。

功能说明：
1. 【动态曝光修正】使用自适应 Gamma 曲线，拯救过曝/欠曝，让画面通透。
2. 【高动态范围】运用 CLAHE (限制对比度自适应直方图均衡化) 揭示暗部隐蔽细节。
3. 【专业级USM锐化】边缘增强，解决画面灰蒙蒙或镜头模糊。
4. 【纯净去噪】"strong" 模式自带彩色非局部均值滤波，去噪除雾。
5. 处理后自动上传，返回全新优化的图片 URL！

使用场景：
- 照片偏暗、逆光、灰蒙蒙（霾天）、或色彩干瘪
- 需要快速提升手机废片的视觉效果

参数说明：
- image_url (必需): 原图片的 URL 地址
- enhance_level (可选): "light"(轻微), "medium"(适中，默认), "strong"(强力去噪+强对比)
- space_id (可选): 目标空间ID
- name (可选): 增强后图片的名称
- introduction (可选): 图片描述
""")
def enhance_and_upload(
    image_url: str,
    enhance_level: str = "medium",
    space_id: Optional[int] = None,
    name: Optional[str] = None,
    introduction: Optional[str] = None
) -> str:
    """执行专业级画质增强并上传到服务器"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证，无法上传。"}'
    
    try:
        knowledge_logger.info(f'[IMAGE_ENHANCE] 开始画质增强 | 级别: {enhance_level} | URL: {image_url}')
        if not image_url: return "请提供图片 URL。"
        
        if enhance_level not in ["light", "medium", "strong"]: enhance_level = "medium"
        
        # 下载图片
        config = load_tool_config()
        headers = {'User-Agent': 'Mozilla/5.0'}
        yuemu = config.get('yuemu', {})
        if yuemu.get('cdn_domain', 'static.yuemutuku.com') in image_url:
            headers['Referer'] = yuemu.get('website_url', 'https://www.yuemutuku.com')
        
        session = _get_requests_session()
        response = session.get(image_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        pil_image = Image.open(BytesIO(response.content))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = image_bgr.shape[:2]
        
        # 增强核心
        enhanced_bgr = enhance_image_quality(image_bgr, enhance_level)
        
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        enhanced_pil = Image.fromarray(enhanced_rgb)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            enhanced_pil.save(tmp_path, 'JPEG', quality=95)
            
            # 上传到 Java 后端
            try:
                from model.factory import load_config
                java_base_url = load_config().get("java_backend_url", "http://127.0.0.1:8123/api")
            except Exception:
                java_base_url = "http://127.0.0.1:8123/api"
            
            upload_url = f"{java_base_url}/picture/upload/postimage"
            upload_headers = {"satoken": token}
            
            params = {}
            if space_id is not None: params['spaceId'] = space_id
            if name: params['picName'] = name
            if introduction: params['introduction'] = introduction
            
            # 引入公共工具并追加图片天然宽高及主色调，确保入库信息完整
            from utils.image_utils import analyze_local_image_attributes
            attrs = analyze_local_image_attributes(tmp_path)
            if attrs:
                params['picWidth'] = attrs['picWidth']
                params['picHeight'] = attrs['picHeight']
                params['picScale'] = attrs['picScale']
                params['picColor'] = attrs['picColor']
            
            with open(tmp_path, 'rb') as f:
                files = {'file': (f'enhanced_{enhance_level}.jpg', f, 'image/jpeg')}
                upload_response = session.post(upload_url, headers=upload_headers, params=params, files=files, timeout=60)
                upload_result = upload_response.json()
                
            if upload_response.status_code == 200 and upload_result.get("code") == 0:
                data = upload_result.get("data", {})
                
                optimizations = ["动态 Gamma 曝光修复", "CLAHE 局域暗部强化", "高级饱和度渲染", "USM 专业边缘锐化"]
                if enhance_level == "strong": optimizations.append("非局部均值去噪降噪")
                
                level_map = {"light": "轻度通透", "medium": "中度质感", "strong": "强力除霾焕新"}
                
                report = ["**图片画质修整完成**\n\n"]
                report.append(f"**处理规格**: {orig_w} × {orig_h} px ({level_map.get(enhance_level)})\n")
                report.append(f"**赋能管线**:\n")
                for opt in optimizations: report.append(f"- {opt}\n")
                report.append(f"\n图片已存入图库，您可以点击图片查看处理后的细节。")
                
                result_data = {
                    "type": "image_enhanced",
                    "msg": "".join(report),
                    "url": data.get("url"),
                    "thumbnailUrl": data.get("thumbnailUrl"),
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
                    "enhance_info": {
                        "original_size": f"{orig_w}x{orig_h}",
                        "enhance_level": enhance_level,
                        "optimizations": optimizations
                    }
                }
                return json.dumps(result_data, ensure_ascii=False)
            else:
                return f'{{"error": "上传增强图片失败: {upload_result.get("message", "未知")}"}}'
        finally:
            try: os.unlink(tmp_path)
            except: pass
            
    except Exception as e:
        knowledge_logger.error(f'[IMAGE_ENHANCE] 画质增强崩溃: {str(e)}')
        return f'{{"error": "画质增强算法异常: {str(e)}"}}'
