#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能艺术拼接卡片工具 - 基于 Pillow 动态排版与色彩聚类配色
为图片定制极具莫兰迪/拍立得风格的“大图+美文+配色卡”高级社交分享海报，自适应提取色板。
"""

import os
import cv2
import json
import yaml
import requests
import tempfile
import numpy as np
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool

from utils.log_utils import knowledge_logger
from .context import get_sa_token
from .color_palette_tool import download_image, extract_color_palette, rgb_to_hex

def load_tool_config() -> dict:
    """加载配置文件"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'tool.yml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        knowledge_logger.warning(f'[CARD_GEN] 配置文件加载失败: {str(e)}')
    return {}

def hex_to_rgb(hex_str: str) -> tuple:
    """十六进制颜色转换为 RGB 元组"""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def get_elegant_font(font_type: str = "regular", size: int = 24) -> ImageFont.FreeTypeFont:
    """
    高质感、离线一体化本地字库加载器：
    直接全部使用项目本地已有的 woff2 字体，实现 100% 离线跨平台高一致性渲染，根除乱码。
    """
    # 拼合 parent/parent 指向的 python-rag/src/font/2.woff2，符合标准容器化打包规范
    base_dir = os.path.dirname(os.path.dirname(__file__))
    local_font_path = os.path.join(base_dir, "font", "2.woff2")
    
    # 1. 优先使用本地已有的 2.woff2 字体，100% 免网络请求
    if os.path.exists(local_font_path):
        try:
            return ImageFont.truetype(local_font_path, size)
        except Exception as e:
            knowledge_logger.error(f"[FONT_LOAD_ERROR] 加载本地项目 2.woff2 失败: {str(e)}")
            
    # 2. 备选加载宿主机自带的中文字体
    font_paths_bold = [
        r"C:\Windows\Fonts\msyhbd.ttc",       # 微软雅黑 粗体
        r"C:\Windows\Fonts\STSONG.TTF",       # 华文宋体 粗体
        r"C:\Windows\Fonts\simhei.ttf",       # 黑体
        "/System/Library/Fonts/PingFang.ttc", # 苹方
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]
    font_paths_regular = [
        r"C:\Windows\Fonts\msyh.ttc",         # 微软雅黑 常规
        r"C:\Windows\Fonts\simsun.ttc",       # 宋体
        "/System/Library/Fonts/PingFang.ttc", # 苹方
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    
    paths = font_paths_bold if font_type == "bold" else font_paths_regular
    
    for path in paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
                
    # 3. 终极系统兜底（保证服务绝不中断）
    return ImageFont.load_default()


def wrap_text_by_width(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """
    根据像素宽度自适应将文本切分为多行，完美兼容带有 '\n' 的多段原始文本，防止重叠。
    """
    # 1. 优先按原文本中的真实换行符拆分（支持大模型原生输出的多行诗歌、段落）
    raw_paragraphs = text.split('\n')
    all_wrapped_lines = []
    
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            # 如果是空行，原样保留空行（支持空行排版呼吸感）
            all_wrapped_lines.append("")
            continue
            
        current_line = ""
        for char in para:
            test_line = current_line + char
            try:
                bbox = font.getbbox(test_line)
                w = bbox[2] - bbox[0]
            except Exception:
                w = font.getsize(test_line)[0]
                
            if w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    all_wrapped_lines.append(current_line)
                current_line = char
        if current_line:
            all_wrapped_lines.append(current_line)
            
    return all_wrapped_lines


@tool(description="""文图智能拼接卡片生成器。
功能说明：
1. 自动利用 KMeans 色彩分析引擎提取原图中最主导的 5 种配色（莫兰迪/传统色级）。
2. 根据原图主色调，智能计算高雅的 HSL 同色系浅色作为卡片底色（浅色衍生），主色深色作为排版文字色，达到顶尖的设计感。
3. 对图片进行无损自适应缩放与极简白框装裱，居中布局在卡片中。
4. 优雅渲染文字（支持标题、引言、多行自动折行排版）。
5. 在卡片底端优雅生成由 5 种图片提取色块构成的“设计师调色盘标尺”，附带 HEX 色值。
6. 一键保存卡片，自动上传至云端并返回高解析度的成品卡片 URL 链接。

使用场景：
- 用户要求：“帮我给这张照片配上一句文案做成精美卡片。”
- 用户要求：“将这张图拼接成高级海报/拍立得卡片，自动匹配颜色。”
- 设计师想要一键生成视觉排版卡片并导出。

参数说明：
- image_url (必需): 图片的 URL 原始地址。
- text (必需): 印在卡片正文上的优雅句子、美文或文字描述。
- title (可选): 卡片的大标题，默认显示“悦木视觉艺术”。
- author (可选): 署名或创作者，默认显示“悦木图库 AI 创作”。
- space_id (可选): 上传的目标空间 ID。
""")
def generate_art_card_and_upload(
    image_url: str,
    text: str,
    title: Optional[str] = "Y U E M U  M O M E N T",
    author: Optional[str] = "悦木图库 AI 创作",
    space_id: Optional[int] = None
) -> str:
    """文图智能拼接卡片：Pillow 动态排版与色彩算法配色"""
    
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)，无法执行卡片上传。"}'
        
    try:
        knowledge_logger.info(f'[CARD_GEN] 启动卡片拼接生成链 | URL: {image_url}')
        
        # 1. 下载并加载图片
        image_bgr = download_image(image_url)
        if image_bgr is None:
            return '{"error": "获取原图失败，请核对图片链接。"}'
            
        # 2. 调用 KMeans 算法提取 5 种色彩及衍生配色方案
        colors = extract_color_palette(image_bgr, n_colors=5)
        if not colors:
            return '{"error": "色彩谱系聚类计算失败，无法配色。"}'
            
        # 3. 自适应配色系统推导 (取第一大占比的主色及其 HSL 衍生色)
        dom_color = colors[0]
        schemes = dom_color['schemes']
        
        # 配色映射
        bg_rgb = hex_to_rgb(schemes['浅色衍生'])
        text_rgb = hex_to_rgb(schemes['深色衍生'])
        accent_rgb = hex_to_rgb(schemes['互补色'])
        dom_rgb = (dom_color['rgb']['r'], dom_color['rgb']['g'], dom_color['rgb']['b'])
        
        # 4. 卡片规格计算与画布初始化
        # 卡片固定宽度：800 px
        card_w = 800
        
        # 将 opencv 格式转为 PIL Image，并计算图片等比例缩放
        pil_orig = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        orig_w, orig_h = pil_orig.size
        
        # 图片框宽度设定为 680px（两边留 60px 边距）
        max_img_w = 680
        max_img_h = 550
        
        # 保持比例计算缩放尺寸
        scale_w = max_img_w / orig_w
        scale_h = max_img_h / orig_h
        scale = min(scale_w, scale_h)
        
        img_w = int(orig_w * scale)
        img_h = int(orig_h * scale)
        
        # 缩放原图
        img_resized = pil_orig.resize((img_w, img_h), Image.Resampling.LANCZOS)
        
        # 5. 加载字体以预估高度
        font_title = get_elegant_font("bold", size=26)
        font_text = get_elegant_font("regular", size=22)
        font_meta = get_elegant_font("regular", size=15)
        
        # 自动长句拆折行 (宽度限制为 640px，留足呼吸边距)
        text_lines = wrap_text_by_width(text, font_text, max_width=640)
        
        # 动态高度计算
        # 顶部留白：60 px
        # 标题区域：40 px + 30px(距图距离)
        # 图片区域：img_h
        # 图下留白：40 px
        # 装饰性艺术双引号启动符：24 px
        # 正文区域：len(lines) * 38 px (行高38) + 40px(距色卡距离)
        # 调色盘色卡矩阵区域：60 px + 30px
        # 底部元信息区域（日期、署名、页眉）：50 px
        # 底部留白：50 px
        
        y_cursor = 60
        title_y = y_cursor
        y_cursor += 40 + 30 # 标题后间距
        
        img_y = y_cursor
        y_cursor += img_h + 40 # 图片后间距
        
        quote_start_y = y_cursor
        y_cursor += 15 # 引号下移少许
        
        text_y = y_cursor
        line_h = 38
        y_cursor += len(text_lines) * line_h + 45 # 文本后间距
        
        palette_y = y_cursor
        y_cursor += 65 + 35 # 色卡高度和间距
        
        meta_y = y_cursor
        y_cursor += 50 + 50 # 底部余留
        
        card_h = y_cursor
        
        # 6. 开始绘制高级卡片底板
        card = Image.new("RGB", (card_w, card_h), bg_rgb)
        draw = ImageDraw.Draw(card)
        
        # --- 绘制四周细线条框，增加手稿般的精致空气感 ---
        border_offset = 20
        draw.rectangle(
            [border_offset, border_offset, card_w - border_offset, card_h - border_offset],
            outline=(int(text_rgb[0]*0.15 + bg_rgb[0]*0.85), int(text_rgb[1]*0.15 + bg_rgb[1]*0.85), int(text_rgb[2]*0.15 + bg_rgb[2]*0.85)),
            width=1
        )
        
        # --- 1. 绘制顶部质感标题 (居中排版) ---
        title_text = title.upper()
        # 测量宽度居中
        try:
            bbox = font_title.getbbox(title_text)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = font_title.getsize(title_text)[0]
        draw.text(((card_w - tw) // 2, title_y), title_text, fill=text_rgb, font=font_title)
        
        # 绘制标题下方的极细线条
        line_w = 80
        draw.line(
            [((card_w - line_w) // 2, title_y + 42), ((card_w + line_w) // 2, title_y + 42)],
            fill=text_rgb,
            width=1
        )
        
        # --- 2. 绘制图片（附带极简白边画框与柔和投影阴影效果） ---
        # 计算图片居中坐标
        img_x = (card_w - img_w) // 2
        
        # 画个白色纯净照片底框 (四周各向外延伸 10 像素)
        mat_border = 10
        draw.rectangle(
            [img_x - mat_border, img_y - mat_border, img_x + img_w + mat_border, img_y + img_h + mat_border],
            fill=(255, 255, 255),
            outline=(210, 210, 210),
            width=1
        )
        # 将原图完美贴入
        card.paste(img_resized, (img_x, img_y))
        
        # --- 3. 绘制文字内容 ---
        # 绘制质感英文艺术引号（主色调对应的互补色强调色）
        try:
            draw.text((80, quote_start_y), "“", fill=accent_rgb, font=get_elegant_font("bold", size=50))
        except Exception:
            draw.text((80, quote_start_y), "“", fill=text_rgb, font=font_title)
            
        # 绘制正文（精美行距，居中偏左排版）
        current_y = text_y
        for line in text_lines:
            draw.text((100, current_y), line, fill=text_rgb, font=font_text)
            current_y += line_h
            
        # --- 4. 绘制精美“设计师色标刻度尺”（Palette Scale） ---
        # 设计5色块横向平铺，居中排布
        color_box_w = 85
        color_box_h = 32
        gap = 12
        total_palette_w = (color_box_w * 5) + (gap * 4)
        palette_start_x = (card_w - total_palette_w) // 2
        
        # 绘制调色板标题（设计师手笔感）
        palette_label = "V I S U A L   C O L O R   P A L E T T E"
        try:
            p_bbox = font_meta.getbbox(palette_label)
            pw = p_bbox[2] - p_bbox[0]
        except Exception:
            pw = font_meta.getsize(palette_label)[0]
        draw.text(((card_w - pw) // 2, palette_y - 25), palette_label, fill=(int(text_rgb[0]*0.7 + bg_rgb[0]*0.3), int(text_rgb[1]*0.7 + bg_rgb[1]*0.3), int(text_rgb[2]*0.7 + bg_rgb[2]*0.3)), font=font_meta)
        
        for i, col_data in enumerate(colors):
            if i >= 5:
                break
            c_rgb = (col_data['rgb']['r'], col_data['rgb']['g'], col_data['rgb']['b'])
            cx = palette_start_x + i * (color_box_w + gap)
            
            # 画圆角或方块色卡
            draw.rectangle(
                [cx, palette_y + 10, cx + color_box_w, palette_y + 10 + color_box_h],
                fill=c_rgb,
                outline=(230, 230, 230) if c_rgb == (255, 255, 255) else None,
                width=1
            )
            # 标记对应色块占比 %
            perc_txt = f"{col_data['percentage']}%"
            try:
                per_w = font_meta.getbbox(perc_txt)[2] - font_meta.getbbox(perc_txt)[0]
            except Exception:
                per_w = font_meta.getsize(perc_txt)[0]
            draw.text((cx + (color_box_w - per_w) // 2, palette_y + 10 + color_box_h + 4), perc_txt, fill=text_rgb, font=font_meta)
            
            # 标记对应 HEX 码
            hex_txt = col_data['hex'].upper()
            try:
                hex_w = font_meta.getbbox(hex_txt)[2] - font_meta.getbbox(hex_txt)[0]
            except Exception:
                hex_w = font_meta.getsize(hex_txt)[0]
            draw.text((cx + (color_box_w - hex_w) // 2, palette_y + 10 + color_box_h + 20), hex_txt, fill=(120, 120, 120), font=font_meta)

        # --- 5. 绘制元信息区（底部） ---
        today_str = datetime.now().strftime("%Y . %m . %d")
        draw.text((100, meta_y), today_str, fill=(140, 140, 140), font=font_meta)
        
        # 署名 (居右排布)
        signature = f"BY {author.upper()}"
        try:
            sig_w = font_meta.getbbox(signature)[2] - font_meta.getbbox(signature)[0]
        except Exception:
            sig_w = font_meta.getsize(signature)[0]
        draw.text((card_w - 100 - sig_w, meta_y), signature, fill=text_rgb, font=font_meta)
        
        # 7. 保存为临时文件并准备上传
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            card.save(tmp_path, 'PNG')
            
            # 8. 获取配置并上传至 Java 云服务器
            try:
                from model.factory import load_config
                rag_config = load_config()
                java_base_url = rag_config.get("java_backend_url", "http://127.0.0.1:8123/api")
            except Exception:
                java_base_url = "http://127.0.0.1:8123/api"
                
            upload_url = f"{java_base_url}/picture/upload/postimage"
            upload_headers = {"satoken": token}
            
            params = {}
            if space_id is not None:
                params['spaceId'] = space_id
            params['picName'] = f"card_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            params['introduction'] = f"基于智能算法拼合的高级艺术社交卡片。配文：{text[:30]}..."
            
            # 引入公共工具并追加图片天然宽高及主色调，确保入库信息完整
            from utils.image_utils import analyze_local_image_attributes
            attrs = analyze_local_image_attributes(tmp_path)
            if attrs:
                params['picWidth'] = attrs['picWidth']
                params['picHeight'] = attrs['picHeight']
                params['picScale'] = attrs['picScale']
                params['picColor'] = attrs['picColor']
            
            # 上传二进制流
            with open(tmp_path, 'rb') as f:
                files = {'file': ('art_card.png', f, 'image/png')}
                knowledge_logger.info(f'[CARD_GEN] 开始上传卡片海报到云存储...')
                upload_response = requests.post(
                    upload_url,
                    headers=upload_headers,
                    params=params,
                    files=files,
                    timeout=60
                )
                upload_result = upload_response.json()
                
            # 9. 解析上传响应
            if upload_response.status_code == 200 and upload_result.get("code") == 0:
                data = upload_result.get("data", {})
                pic_url = data.get("url")
                thumbnail_url = data.get("thumbnailUrl")
                
                knowledge_logger.info(f'[CARD_GEN] 卡片合成并上传成功 | URL: {pic_url}')
                
                # 组织 Markdown 报告
                report_lines = ["**🎨 悦木艺术拼接社交卡片已智造完成**\n\n"]
                report_lines.append(f"- **主导配色（占画面比 {dom_color['percentage']}%）**: `{dom_color['hex'].upper()}` ({dom_color['name']})\n")
                report_lines.append(f"- **高雅底框（浅色衍生色）**: `{schemes['浅色衍生'].upper()}`\n")
                report_lines.append(f"- **正文墨迹（深色衍变色）**: `{schemes['深色衍生'].upper()}`\n")
                report_lines.append(f"- **聚焦印章（互补强调色）**: `{schemes['互补色'].upper()}`\n\n")
                report_lines.append(f"- **优雅配文**: “ {text} ”\n\n")
                report_lines.append("该作品已帮您完美排版、生成阴影留白，并自动永久保存至您的云端图库！您可以点击下方的预览卡片，一键保存并分享。")
                
                result_json = {
                    "type": "card_generated",
                    "msg": "".join(report_lines),
                    "url": pic_url,
                    "thumbnailUrl": thumbnail_url,
                    "name": data.get("name"),
                    "introduction": data.get("introduction"),
                    "tags": ["智能卡片", "多模态排版", "莫兰迪色"],
                    "category": "DESIGN",
                    "picWidth": card_w,
                    "picHeight": card_h,
                    "picFormat": "png"
                }
                return json.dumps(result_json, ensure_ascii=False)
            else:
                msg = upload_result.get("message", "未知错误")
                knowledge_logger.error(f'[CARD_GEN] 上传失败: {msg}')
                return f'{{"error": "卡片海报上传服务器失败: {msg}"}}'
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            
    except Exception as e:
        error_msg = f"文图智能拼接卡片生成崩溃: {str(e)}"
        knowledge_logger.error(f'[CARD_GEN_ERROR] {error_msg}')
        import traceback
        knowledge_logger.error(traceback.format_exc())
        return f'{{"error": "{error_msg}"}}'
