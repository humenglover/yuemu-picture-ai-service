#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
©️ 智能水印工具 (Automated Watermarking) - V2 (专业版权保护版)
支持自定义样式、防搬运强力平铺、EXIF版权写入及全方位异常兼容。
"""
import os
import cv2
import yaml
import json
import requests
import tempfile
import math
import numpy as np
import piexif
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from typing import Optional
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger
from .context import get_sa_token

def load_tool_config() -> dict:
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'tool.yml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        knowledge_logger.warning(f'[WATERMARK] 加载配置文件失败: {str(e)}')
    return {}

def hex_to_rgba(hex_color: str, opacity: int) -> tuple:
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (opacity,)
    return (255, 255, 255, opacity)

def add_visible_watermark(
    image: Image.Image,
    text: str,
    position: str = "bottom-right",
    opacity: int = 128,
    color: str = "#FFFFFF",
    angle: int = 0,
    size_ratio: float = 0.05
) -> Image.Image:
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
        
    watermark_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    # 字体自适应（完美加载本地已有的 2.woff2 离线中文字体，彻底根除 Linux 乱码）
    font_size = max(12, int(min(image.width, image.height) * size_ratio))
    font = None
    
    # 拼合 parent/parent 指向的 python-rag/src/font/2.woff2，符合标准容器化打包规范
    base_dir = os.path.dirname(os.path.dirname(__file__))
    local_font_path = os.path.join(base_dir, "font", "2.woff2")
    
    # 1. 优先使用本地已有的 2.woff2 字体，100% 免网络请求
    if os.path.exists(local_font_path):
        try:
            font = ImageFont.truetype(local_font_path, font_size)
        except Exception as e:
            knowledge_logger.error(f"[FONT_LOAD_ERROR] 加载本地项目 2.woff2 水印字体失败: {str(e)}")
            
    # 2. 备选加载操作系统预置中文字体
    if font is None:
        try:
            font_paths = [
                "C:\\Windows\\Fonts\\msyh.ttc",
                "C:\\Windows\\Fonts\\simhei.ttf",
                "/System/Library/Fonts/PingFang.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # 英文兜底
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
        except Exception:
            pass
            
    # 3. 终极系统物理兜底
    if font is None:
        font = ImageFont.load_default()
        
    watermark_color = hex_to_rgba(color, opacity)
    
    # 获取文本尺寸并创建一个刚好包围文本的图层
    temp_draw = ImageDraw.Draw(watermark_layer)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    left, top, right, bottom = bbox
    text_width = right - left
    text_height = bottom - top
    padding = max(10, font_size // 2)
    
    # 使用与水印相同的RGB但完全透明的背景，防止旋转或插值时出现黑边
    bg_color = (watermark_color[0], watermark_color[1], watermark_color[2], 0)
    txt_layer = Image.new('RGBA', (text_width + padding * 2, text_height + padding * 2), bg_color)
    txt_draw = ImageDraw.Draw(txt_layer)
    # 减去 left 和 top 的偏移量，保证文字完全在图层内
    txt_draw.text((padding - left, padding - top), text, font=font, fill=watermark_color)
    
    # 倾斜处理
    if angle != 0:
        txt_layer = txt_layer.rotate(angle, expand=True, resample=Image.BICUBIC)
        text_width, text_height = txt_layer.size
    else:
        text_width, text_height = txt_layer.size
        
    if position == "tile":
        # 强力平铺（防搬运）
        spacing_x = text_width + max(50, int(image.width * 0.1))
        spacing_y = text_height + max(50, int(image.height * 0.1))
        for y in range(-text_height, image.height, spacing_y):
            for x in range(-text_width, image.width, spacing_x):
                watermark_layer.paste(txt_layer, (x, y), txt_layer)
    else:
        margin = max(10, int(min(image.width, image.height) * 0.02))
        if position == "bottom-right":
            x = image.width - text_width - margin
            y = image.height - text_height - margin
        elif position == "bottom-left":
            x = margin
            y = image.height - text_height - margin
        elif position == "top-right":
            x = image.width - text_width - margin
            y = margin
        elif position == "top-left":
            x = margin
            y = margin
        elif position == "center":
            x = (image.width - text_width) // 2
            y = (image.height - text_height) // 2
        else:
            x = image.width - text_width - margin
            y = image.height - text_height - margin
            
        watermark_layer.paste(txt_layer, (x, y), txt_layer)
        
    watermarked = Image.alpha_composite(image, watermark_layer)
    if watermarked.mode == 'RGBA':
        rgb_image = Image.new('RGB', watermarked.size, (255, 255, 255))
        rgb_image.paste(watermarked, mask=watermarked.split()[3])
        return rgb_image
    return watermarked

def add_blind_watermark(image_bgr: np.ndarray, text: str) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    watermark = np.zeros((h, w), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(h, w) / 500.0
    thickness = max(1, int(font_scale * 2))
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - text_width) // 2
    y = (h + text_height) // 2
    cv2.putText(watermark, text, (x, y), font, font_scale, 255, thickness)
    
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    watermark_strength = 0.01
    f_shift_watermarked = f_shift + watermark_strength * watermark
    f_ishift = np.fft.ifftshift(f_shift_watermarked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    watermarked_bgr = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
    return watermarked_bgr

def write_exif_copyright(image_path: str, copyright_text: str):
    """写入 EXIF 版权信息（版权号、作者等），支持专业版权保护"""
    try:
        from piexif import helper
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        try:
            exif_dict = piexif.load(image_path)
        except Exception:
            pass # 可能没有exif
        
        exif_dict['0th'][piexif.ImageIFD.Copyright] = copyright_text.encode('utf-8')
        exif_dict['0th'][piexif.ImageIFD.Artist] = copyright_text.encode('utf-8')
        exif_dict['Exif'][piexif.ExifIFD.UserComment] = helper.UserComment.dump(copyright_text, encoding="unicode")
        
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)
        knowledge_logger.info(f"[WATERMARK] 已成功写入 EXIF 版权信息: {copyright_text}")
    except Exception as e:
        knowledge_logger.warning(f"[WATERMARK] EXIF写入失败 (可能不支持该图片格式): {e}")

@tool(description="""专业图片水印工具，支持强力平铺、防搬运、旋转、透明度、字体颜色定制及 EXIF 版权写入。

参数说明：
- image_url (必需): 原图片的 URL 地址
- watermark_text (必需): 水印文本内容
- watermark_type (可选): "visible"（明文，默认）或 "blind"（盲水印）
- position (可选): 水印位置，"tile"（平铺/全屏斜纹，推荐自媒体防搬运）, "bottom-right"等。
- opacity (可选): 透明度，0-255，默认 128
- color (可选): 水印颜色，HEX格式，如 "#FFFFFF" 或 "#FF0000"
- angle (可选): 旋转角度，支持斜角水印（如 45）
- size_ratio (可选): 水印大小比例，默认 0.05（自适应）
- write_exif (可选): 是否写入EXIF底层版权信息，默认 True
- space_id (可选): 上传空间ID
- name (可选): 存储文件名
- introduction (可选): 图片描述
""")
def add_watermark_and_upload(
    image_url: str,
    watermark_text: str,
    watermark_type: str = "visible",
    position: str = "bottom-right",
    opacity: int = 128,
    color: str = "#FFFFFF",
    angle: int = 0,
    size_ratio: float = 0.05,
    write_exif: bool = True,
    space_id: Optional[int] = None,
    name: Optional[str] = None,
    introduction: Optional[str] = None
) -> str:
    """为图片添加专业水印并上传"""
    
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)。"}'
    
    try:
        knowledge_logger.info(f'[WATERMARK] 处理开始 | URL: {image_url} | 文本: {watermark_text}')
        if not image_url or not watermark_text:
            return "请提供有效的图片 URL 和 水印文本。"
            
        watermark_type = "visible" if watermark_type not in ["visible", "blind"] else watermark_type
        opacity = max(0, min(255, int(opacity)))
        angle = int(angle)
        size_ratio = max(0.01, min(1.0, float(size_ratio)))
        
        # 1. 下载并兼容不同图片格式 (GIF/WebP/PNG)
        config = load_tool_config()
        headers = {'User-Agent': 'Mozilla/5.0'}
        yuemu = config.get('yuemu', {})
        if yuemu.get('cdn_domain', 'static.yuemutuku.com') in image_url:
            headers['Referer'] = yuemu.get('website_url', 'https://www.yuemutuku.com')
            
        response = requests.get(image_url, headers=headers, timeout=20)
        response.raise_for_status()
        
        pil_image = Image.open(BytesIO(response.content))
        
        # 如果是GIF，取第一帧
        if getattr(pil_image, "is_animated", False):
            pil_image.seek(0)
            
        # 尺寸限制（过大图片避免 OOM，最大支持 8000x8000）
        orig_w, orig_h = pil_image.size
        max_dim = 8000
        if orig_w > max_dim or orig_h > max_dim:
            pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
        # 转为安全格式
        if pil_image.mode not in ['RGB', 'RGBA']:
            pil_image = pil_image.convert('RGBA')
            
        knowledge_logger.info(f'[WATERMARK] 图像加载成功: {orig_w}x{orig_h}')
        
        # 2. 添加水印
        if watermark_type == "visible":
            watermarked_pil = add_visible_watermark(
                pil_image, watermark_text, position, opacity, color, angle, size_ratio
            )
            desc = f"明文水印（位置:{position}, 透明度:{opacity}, 角度:{angle}, 颜色:{color}）"
        else:
            # 盲水印不支持透明通道，强制转 RGB
            if pil_image.mode == 'RGBA':
                rgb_bg = Image.new('RGB', pil_image.size, (255, 255, 255))
                rgb_bg.paste(pil_image, mask=pil_image.split()[3])
                pil_image = rgb_bg
            else:
                pil_image = pil_image.convert('RGB')
                
            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            watermarked_bgr = add_blind_watermark(image_bgr, watermark_text)
            watermarked_rgb = cv2.cvtColor(watermarked_bgr, cv2.COLOR_BGR2RGB)
            watermarked_pil = Image.fromarray(watermarked_rgb)
            desc = "隐形盲水印（FFT 频域）"
            
        # 3. 保存临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # JPEG 不支持 RGBA
            if watermarked_pil.mode == 'RGBA':
                rgb_image = Image.new('RGB', watermarked_pil.size, (255, 255, 255))
                rgb_image.paste(watermarked_pil, mask=watermarked_pil.split()[3])
                watermarked_pil = rgb_image
                
            watermarked_pil.save(tmp_path, 'JPEG', quality=95)
            
            # 写入 EXIF
            if write_exif:
                write_exif_copyright(tmp_path, watermark_text)
                desc += " [带有强力 EXIF 版权保护]"
                
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
                files = {'file': (f'watermark_{watermark_type}.jpg', f, 'image/jpeg')}
                upload_response = requests.post(upload_url, headers=upload_headers, params=params, files=files, timeout=60)
                upload_result = upload_response.json()
                
            if upload_response.status_code == 200 and upload_result.get("code") == 0:
                data = upload_result.get("data", {})
                pic_url = data.get("url")
                
                report = (
                    f"©️ **专业水印添加完成**\n\n"
                    f"- **尺寸**: {orig_w}x{orig_h}\n"
                    f"- **水印配置**: {desc}\n"
                    f"- **内容**: {watermark_text}\n"
                    f"- **EXIF 写入**: {'成功' if write_exif else '跳过'}\n"
                )
                
                result_data = {
                    "type": "image_watermarked",
                    "msg": report,
                    "url": pic_url,
                    "thumbnailUrl": data.get("thumbnailUrl"),
                    "name": data.get("name"),
                    "picFormat": data.get("picFormat"),
                    "watermark_info": {
                        "text": watermark_text,
                        "position": position,
                        "angle": angle,
                        "color": color,
                        "exif_protected": write_exif
                    }
                }
                return json.dumps(result_data, ensure_ascii=False)
            else:
                return f'{{"error": "上传失败: {upload_result.get("message", "未知错误")}"}}'
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
            
    except Exception as e:
        knowledge_logger.error(f'[WATERMARK] 处理异常: {str(e)}')
        return f'{{"error": "水印处理发生错误: {str(e)}"}}'
