#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
艺术装裱工具 (Image Framing Tool)
支持将图片进行虚拟装裱（木框、金框、极简卡纸、中式画框等），
通过高精度的 Pillow 绘图和阴影模拟，实现极富质感的艺术展示效果。
"""
import os
import cv2
import yaml
import json
import requests
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps, ImageDraw, ImageFilter
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
        knowledge_logger.warning(f'[FRAME] 加载配置文件失败: {str(e)}')
    return {}

def draw_art_frame(img: Image.Image, frame_style: str) -> Image.Image:
    """对 PIL 图像进行艺术装裱处理"""
    frame_style = frame_style.lower().strip()
    W, H = img.size
    
    # 1. 自适应计算边框尺寸
    base_pad = max(24, int(min(W, H) * 0.08))  # 卡纸内衬宽度 (卡纸)
    
    if frame_style == "matte_white":
        # 极简白卡纸：白色宽卡纸 + 细浅灰内线 + 软阴影 + 纤细极简黑外边框
        # 先在原图周围加上浅浅的 1px 细缝 (边缘间隙)
        img_with_gap = ImageOps.expand(img, border=2, fill=(245, 245, 245))
        
        # 创建大画布（放置卡纸）
        matte_w = W + 4 + base_pad * 2
        matte_h = H + 4 + base_pad * 2
        matte = Image.new('RGB', (matte_w, matte_h), (252, 251, 249)) # 略带米白的高级美术纸质感
        
        # 给原图生成软阴影 (Soft Shadow)
        shadow_pad = base_pad // 4
        shadow = Image.new('RGBA', (W + shadow_pad * 2, H + shadow_pad * 2), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        # 画半透明黑色圆角阴影
        shadow_draw.rectangle([shadow_pad, shadow_pad, W + shadow_pad, H + shadow_pad], fill=(0, 0, 0, 35))
        shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_pad // 2))
        
        # 合成阴影和原图到卡纸上
        matte.paste(shadow, (base_pad - shadow_pad + 2, base_pad - shadow_pad + 2), shadow)
        matte.paste(img_with_gap, (base_pad, base_pad))
        
        # 在卡纸内侧绘制一圈纤细的凹槽线 (Matte line)
        draw = ImageDraw.Draw(matte)
        line_offset = base_pad // 3
        draw.rectangle(
            [line_offset, line_offset, matte_w - line_offset, matte_h - line_offset],
            outline=(210, 208, 204), width=1
        )
        # 最外层加上纤细的黑外框 (1px)
        framed = ImageOps.expand(matte, border=1, fill=(20, 20, 20))
        return framed
        
    elif frame_style == "matte_black":
        # 现代黑卡纸：深邃纯黑卡纸 + 细深灰凹槽线 + 纤细纯白外边框
        img_with_gap = ImageOps.expand(img, border=2, fill=(30, 30, 30))
        matte_w = W + 4 + base_pad * 2
        matte_h = H + 4 + base_pad * 2
        matte = Image.new('RGB', (matte_w, matte_h), (18, 18, 18)) # 纯黑偏哑光美术纸
        
        # 将原图贴入
        matte.paste(img_with_gap, (base_pad, base_pad))
        
        # 绘制黑卡凹槽线 (偏亮灰色)
        draw = ImageDraw.Draw(matte)
        line_offset = base_pad // 3
        draw.rectangle(
            [line_offset, line_offset, matte_w - line_offset, matte_h - line_offset],
            outline=(50, 50, 50), width=1
        )
        # 最外层一圈白外框
        framed = ImageOps.expand(matte, border=2, fill=(230, 230, 230))
        return framed
        
    elif frame_style == "wood":
        # 复古胡桃木框：复古木边框 + 白色卡纸
        border_w = max(10, int(base_pad * 0.4)) # 木框宽度
        
        # 1. 贴白卡纸内衬
        matte_w = W + base_pad * 2
        matte_h = H + base_pad * 2
        matte = Image.new('RGB', (matte_w, matte_h), (250, 248, 243))
        # 贴原图
        matte.paste(img, (base_pad, base_pad))
        # 细凹槽
        draw = ImageDraw.Draw(matte)
        draw.rectangle(
            [base_pad - 10, base_pad - 10, W + base_pad + 10, H + base_pad + 10],
            outline=(220, 218, 212), width=1
        )
        
        # 2. 贴复古红褐胡桃木外边框 (分层描边)
        # 用渐变色来模拟木纹的立体边框 (暗-亮-暗)
        framed = ImageOps.expand(matte, border=border_w, fill=(65, 43, 21)) # 基础深胡桃木色
        draw_wood = ImageDraw.Draw(framed)
        # 内描边高光
        draw_wood.rectangle(
            [border_w - 2, border_w - 2, framed.width - border_w + 2, framed.height - border_w + 2],
            outline=(95, 68, 41), width=2
        )
        # 外描边阴影
        draw_wood.rectangle(
            [0, 0, framed.width - 1, framed.height - 1],
            outline=(35, 23, 11), width=3
        )
        return framed
        
    elif frame_style == "chinese":
        # 传统中式装裱：红木外框 + 经典米黄绢面内边 + 细金内框
        pad_silk = base_pad
        border_w = max(12, int(base_pad * 0.35))
        
        # 1. 创建丝绸绢面
        matte_w = W + pad_silk * 2
        matte_h = H + pad_silk * 2
        matte = Image.new('RGB', (matte_w, matte_h), (242, 231, 201)) # 宣纸米黄/国画绢本黄
        
        # 在原图外贴一圈细金线内衬 (Chinese Gold)
        img_with_gold = ImageOps.expand(img, border=4, fill=(197, 160, 89))
        matte.paste(img_with_gold, (pad_silk - 4, pad_silk - 4))
        
        # 2. 贴红木外框
        framed = ImageOps.expand(matte, border=border_w, fill=(112, 28, 28)) # 中式红木色
        draw_wood = ImageDraw.Draw(framed)
        # 描金外框线 (带金边阴影)
        draw_wood.rectangle(
            [border_w, border_w, framed.width - border_w, framed.height - border_w],
            outline=(153, 118, 59), width=1
        )
        # 最外层阴影
        draw_wood.rectangle(
            [0, 0, framed.width - 1, framed.height - 1],
            outline=(60, 15, 15), width=2
        )
        return framed
        
    elif frame_style == "golden":
        # 奢华欧式金框：金黄立体边框 + 象牙白卡纸
        border_w = max(12, int(base_pad * 0.45))
        
        # 1. 贴卡纸
        matte_w = W + base_pad * 2
        matte_h = H + base_pad * 2
        matte = Image.new('RGB', (matte_w, matte_h), (248, 246, 240)) # 象牙白
        matte.paste(img, (base_pad, base_pad))
        
        # 2. 贴立体金色描边
        framed = ImageOps.expand(matte, border=border_w, fill=(212, 175, 55)) # 奢华金
        draw_gold = ImageDraw.Draw(framed)
        # 贴多条线模拟欧式雕花框的起伏立体感
        draw_gold.rectangle(
            [border_w - 4, border_w - 4, framed.width - border_w + 4, framed.height - border_w + 4],
            outline=(245, 222, 129), width=2  # 内圈高光金
        )
        draw_gold.rectangle(
            [border_w // 2, border_w // 2, framed.width - border_w // 2, framed.height - border_w // 2],
            outline=(166, 124, 25), width=3   # 中圈阴影古铜金
        )
        draw_gold.rectangle(
            [0, 0, framed.width - 1, framed.height - 1],
            outline=(130, 95, 10), width=2    # 最外框纯黑金
        )
        return framed
        
    # 默认：极简白
    return draw_art_frame(img, "matte_white")

@tool(description="""专业图像艺术装裱大师，支持为摄影、画作、插画等一键添加高级画框。

风格支持：
- matte_white: 现代极简白（白色高级美术纸宽内衬、软投影、凹槽线、纤细极简黑外框，最推荐）
- matte_black: 奢雅美术黑（深邃全黑卡纸内衬、深灰线凹槽、纯白描边外框，具有高冷艺术感）
- wood: 复古胡桃木（天然胡桃木褐色粗框、象牙白卡纸内衬，适合摄影、插画）
- chinese: 经典中式装裱（中式红木边框、宣纸米黄绢衬、细金内边，适合国画、书法、水墨）
- golden: 欧式奢华金（欧式宫廷立体雕花描金框、高级卡纸内衬，高贵大气）

参数说明：
- image_url (必需): 待装裱处理的图片 URL 地址
- style (必需): 艺术画框风格，必须是上面 5 种之一："matte_white", "matte_black", "wood", "chinese", "golden"
- space_id (可选): 目标保存空间 ID
- name (可选): 保存的文件名
- introduction (可选): 图片描述
""")
def apply_frame_and_upload(
    image_url: str,
    style: str,
    space_id: Optional[int] = None,
    name: Optional[str] = None,
    introduction: Optional[str] = None
) -> str:
    """给图片添加高级艺术画框装裱并上传"""
    
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)。"}'
        
    try:
        knowledge_logger.info(f'[FRAME] 画框处理开始 | Style: {style} | URL: {image_url}')
        if not image_url or not style:
            return "请提供有效的图片 URL 和 装裱画框风格名称。"
            
        style = style.lower().strip()
        style_map = {
            "matte_white": "现代极简白框装裱",
            "matte_black": "奢雅美术黑框装裱",
            "wood": "复古胡桃木框装裱",
            "chinese": "国画中式红木装裱",
            "golden": "欧式古典金框装裱"
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
            
        # 限制原图最大边长防止 OOM
        orig_w, orig_h = pil_image.size
        max_dim = 4000  # 装裱本身需要增加尺寸，原图限制稍低一些
        if orig_w > max_dim or orig_h > max_dim:
            pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        # 2. 艺术装裱处理
        framed_pil = draw_art_frame(pil_image, style)
        new_w, new_h = framed_pil.size
        
        # 3. 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            framed_pil.save(tmp_path, 'JPEG', quality=95)
            
            # 4. 上传给后端
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
                files = {'file': (f'framed_{style}.jpg', f, 'image/jpeg')}
                upload_response = requests.post(upload_url, headers=upload_headers, params=params, files=files, timeout=60)
                upload_result = upload_response.json()
                
            if upload_response.status_code == 200 and upload_result.get("code") == 0:
                data = upload_result.get("data", {})
                pic_url = data.get("url")
                
                report = (
                    f"**艺术装裱完成**\n\n"
                    f"- **装裱样式**: {style_map[style]} (`{style}`)\n"
                    f"- **装裱后尺寸**: {new_w}x{new_h} px (原图: {orig_w}x{orig_h})\n"
                    f"- **状态**: 已完美装裱，并保存至您的图库空间。\n"
                )
                
                result_data = {
                    "type": "image_framed",
                    "msg": report,
                    "url": pic_url,
                    "thumbnailUrl": data.get("thumbnailUrl"),
                    "name": data.get("name"),
                    "picFormat": data.get("picFormat"),
                    "frame_info": {
                        "style_code": style,
                        "style_name": style_map[style]
                    }
                }
                return json.dumps(result_data, ensure_ascii=False)
            else:
                return f'{{"error": "图片装裱上传失败: {upload_result.get("message", "未知错误")}"}}'
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
            
    except Exception as e:
        knowledge_logger.error(f'[FRAME] 处理异常: {str(e)}')
        return f'{{"error": "图像艺术装裱处理失败: {str(e)}"}}'
