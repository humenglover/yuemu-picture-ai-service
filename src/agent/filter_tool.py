#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
氛围感滤镜工具 (LUTs Filter Tool)
支持赛博朋克、日系清新、胶片复古、黑白质感、暖阳午后等一键风格化处理，
利用 OpenCV/NumPy 的高精度色彩重映射算法，计算速度极快。
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
from typing import Optional
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
        knowledge_logger.warning(f'[FILTER] 加载配置文件失败: {str(e)}')
    return {}

def apply_style_filter(img: np.ndarray, style: str) -> np.ndarray:
    """给 BGR 格式的图像应用不同的艺术色彩滤镜"""
    style = style.lower()
    
    if style == "cyberpunk":
        # 1. 赛博朋克风：冷暖强烈对比，霓虹青与霓虹粉
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b_chan = cv2.split(lab)
        # 偏向洋红(a加深)和冷蓝(b减小)
        a = cv2.add(a, 18)
        b_chan = cv2.subtract(b_chan, 22)
        lab = cv2.merge((l, a, b_chan))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # 拉大暗部与亮部对比度
        img = cv2.convertScaleAbs(img, alpha=1.12, beta=8)
        
    elif style == "japanese":
        # 2. 日系清新风：高曝光、低饱和、微微偏蓝绿冷色
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        h, s, v = cv2.split(hsv)
        # 降低饱和度，呈现淡雅色彩
        s = np.clip(s * 0.72, 0, 255).astype(np.uint8)
        # 提升亮度（高光感）
        v = np.clip(v * 1.15 + 15, 0, 255).astype(np.uint8)
        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)
        # 微调偏色
        b, g, r = cv2.split(img)
        b = np.clip(b * 1.06 + 8, 0, 255).astype(np.uint8)
        g = np.clip(g * 1.02 + 4, 0, 255).astype(np.uint8)
        r = np.clip(r * 0.94, 0, 255).astype(np.uint8)
        img = cv2.merge((b, g, r))
        
    elif style == "retro":
        # 3. 胶片复古风：暖阳微黄、暗部褪色(Fade Black)、胶片微粒噪点
        # 调暖色调（红黄增加，蓝减少）
        b, g, r = cv2.split(img)
        r = np.clip(r * 1.12 + 8, 0, 255).astype(np.uint8)
        g = np.clip(g * 1.04 + 2, 0, 255).astype(np.uint8)
        b = np.clip(b * 0.86, 0, 255).astype(np.uint8)
        img = cv2.merge((b, g, r))
        # 褪色效果（暗部不沉底，灰度拉高）
        img = cv2.convertScaleAbs(img, alpha=0.84, beta=22)
        # 叠加随机胶片微粒（Grains）
        noise = np.random.normal(0, 4.5, img.shape).astype(np.int8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    elif style == "classic_bw":
        # 4. 黑白经典风：并非纯去色，而是利用高对比度展现纪实肖像质感
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高光和暗部的 Sigmoid 非线性拉伸
        lookup_table = np.array([255 / (1 + np.exp(-0.028 * (i - 120))) for i in range(256)]).astype(np.uint8)
        gray = cv2.LUT(gray, lookup_table)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
    elif style == "warm_sunset":
        # 5. 暖阳午后风：浓郁的晚霞橘红，温馨治愈
        b, g, r = cv2.split(img)
        r = np.clip(r * 1.18 + 12, 0, 255).astype(np.uint8)
        g = np.clip(g * 1.06 + 6, 0, 255).astype(np.uint8)
        b = np.clip(b * 0.82, 0, 255).astype(np.uint8)
        img = cv2.merge((b, g, r))
        
    return img

@tool(description="""专业氛围感色彩滤镜大师，支持一键将图片转换为不同的艺术摄影风格。

风格支持：
- cyberpunk: 赛博朋克风（霓虹冷暖强烈对比、科幻未来感）
- japanese: 日系清新风（明亮高曝光、清透低饱和、偏蓝绿冷调）
- retro: 胶片复古风（胶片颗粒感、暗部褪色、温润胶片黄暖调）
- classic_bw: 经典高反差黑白（纪实肖像大师风骨，对比浓烈）
- warm_sunset: 暖阳午后风（橘红色夕阳温馨色调）

参数说明：
- image_url (必需): 待处理的图片 URL 地址
- style (必需): 艺术风格类型，必须是以上 5 种之一："cyberpunk", "japanese", "retro", "classic_bw", "warm_sunset"
- space_id (可选): 目标保存空间 ID
- name (可选): 保存的文件名
- introduction (可选): 图片描述
""")
def apply_filter_and_upload(
    image_url: str,
    style: str,
    space_id: Optional[int] = None,
    name: Optional[str] = None,
    introduction: Optional[str] = None
) -> str:
    """给图片应用艺术氛围感滤镜并上传"""
    
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)。"}'
        
    try:
        knowledge_logger.info(f'[FILTER] 滤镜处理开始 | Style: {style} | URL: {image_url}')
        if not image_url or not style:
            return "请提供有效的图片 URL 和 滤镜风格名称。"
            
        style = style.lower().strip()
        style_map = {
            "cyberpunk": "赛博朋克霓虹风",
            "japanese": "日系治愈清新风",
            "retro": "经典胶片复古风",
            "classic_bw": "高反差黑白纪实风",
            "warm_sunset": "暖阳午后温馨风"
        }
        
        if style not in style_map:
            return f"不支持该风格：'{style}'。请选择：{', '.join(style_map.keys())}。"
            
        # 1. 下载图片并绕过防盗链
        config = load_tool_config()
        headers = {'User-Agent': 'Mozilla/5.0'}
        yuemu = config.get('yuemu', {})
        if yuemu.get('cdn_domain', 'static.yuemutuku.com') in image_url:
            headers['Referer'] = yuemu.get('website_url', 'https://www.yuemutuku.com')
            
        response = requests.get(image_url, headers=headers, timeout=20)
        response.raise_for_status()
        
        pil_image = Image.open(BytesIO(response.content))
        
        # 兼容 GIF/PNG 等
        if getattr(pil_image, "is_animated", False):
            pil_image.seek(0)
            
        # 尺寸自适应防 OOM
        orig_w, orig_h = pil_image.size
        max_dim = 6000
        if orig_w > max_dim or orig_h > max_dim:
            pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        # 2. 转换并处理滤镜
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        filtered_bgr = apply_style_filter(image_bgr, style)
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        filtered_pil = Image.fromarray(filtered_rgb)
        
        # 3. 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            filtered_pil.save(tmp_path, 'JPEG', quality=95)
            
            # 4. 上传
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
                files = {'file': (f'filtered_{style}.jpg', f, 'image/jpeg')}
                upload_response = requests.post(upload_url, headers=upload_headers, params=params, files=files, timeout=60)
                upload_result = upload_response.json()
                
            if upload_response.status_code == 200 and upload_result.get("code") == 0:
                data = upload_result.get("data", {})
                pic_url = data.get("url")
                
                report = (
                    f"**氛围感滤镜生成完成**\n\n"
                    f"- **滤镜风格**: {style_map[style]} (`{style}`)\n"
                    f"- **图片尺寸**: {orig_w}x{orig_h} px\n"
                    f"- **状态**: 一键质感调色已完成，并保存到图库。\n"
                )
                
                result_data = {
                    "type": "image_filtered",
                    "msg": report,
                    "url": pic_url,
                    "thumbnailUrl": data.get("thumbnailUrl"),
                    "name": data.get("name"),
                    "picFormat": data.get("picFormat"),
                    "filter_info": {
                        "style_code": style,
                        "style_name": style_map[style]
                    }
                }
                return json.dumps(result_data, ensure_ascii=False)
            else:
                return f'{{"error": "应用滤镜后上传失败: {upload_result.get("message", "未知错误")}"}}'
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
            
    except Exception as e:
        knowledge_logger.error(f'[FILTER] 处理异常: {str(e)}')
        return f'{{"error": "色彩滤镜处理失败: {str(e)}"}}'
