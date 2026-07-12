#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
色板提取工具 - Pro版 (基于 CIE LAB 感知空间 K-Means 聚类)
增强了 UI/UX 专业配色方案推荐、高级色阶命名和同类色/互补色推导。
"""

import os
import cv2
import yaml
import requests
import colorsys
import math
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Dict, Optional
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger

# -- 高级色彩字典库 (涵盖部分莫兰迪、传统色及常用 Web 色) --
COLOR_DICT = {
    "极致黑": (0, 0, 0), "纯净白": (255, 255, 255),
    "深海蓝": (20, 30, 80), "海军蓝": (0, 0, 128), "宝蓝色": (0, 115, 207), "天蓝色": (135, 206, 235), "马卡龙蓝": (176, 224, 230),
    "中国红": (224, 0, 48), "勃艮第红": (128, 0, 32), "珊瑚红": (255, 127, 80), "莫兰迪粉": (212, 186, 186), "樱花粉": (255, 183, 197),
    "森林绿": (34, 139, 34), "墨绿色": (0, 64, 64), "薄荷绿": (152, 255, 152), "莫兰迪绿": (154, 174, 151), "橄榄绿": (128, 128, 0),
    "赤金色": (222, 118, 34), "向日葵黄": (255, 215, 0), "香槟金": (247, 231, 206), "奶油黄": (253, 253, 150),
    "葡萄紫": (106, 40, 126), "丁香紫": (200, 162, 200), "莫兰迪紫": (150, 138, 160),
    "深咖啡色": (75, 54, 33), "卡其色": (195, 176, 145), "燕麦色": (223, 215, 205),
    "碳灰色": (54, 69, 79), "高级灰": (128, 128, 128), "银灰色": (192, 192, 192)
}

def rgb_to_hex(rgb: tuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_text_color_suggestion(rgb: tuple) -> str:
    """WCAG 亮度推导，判断背景该配黑字还是白字"""
    a = [c / 255.0 for c in rgb]
    a = [(c / 12.92) if c <= 0.03928 else (((c + 0.055) / 1.055) ** 2.4) for c in a]
    luminance = 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2]
    return "黑色" if luminance > 0.179 else "白色"

def get_advanced_color_schemes(rgb: tuple) -> Dict[str, str]:
    """计算多种专业配色方案"""
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
    def _hsl_to_hex(hn, ln, sn):
        cr, cg, cb = colorsys.hls_to_rgb(hn % 1.0, max(0, min(1, ln)), max(0, min(1, sn)))
        return rgb_to_hex((int(cr * 255), int(cg * 255), int(cb * 255)))

    return {
        "互补色": _hsl_to_hex(h + 0.5, l, s),                           # 180度
        "同类色_暖": _hsl_to_hex(h + 0.08, l, s),                        # +30度
        "同类色_冷": _hsl_to_hex(h - 0.08, l, s),                        # -30度
        "三角色_1": _hsl_to_hex(h + 0.333, l, s),                       # +120度
        "三角色_2": _hsl_to_hex(h - 0.333, l, s),                       # -120度
        "浅色衍生": _hsl_to_hex(h, min(0.9, l + 0.2), s),              # 同色相高明度 (适合背景)
        "深色衍生": _hsl_to_hex(h, max(0.1, l - 0.3), s)               # 同色相低明度 (适合描边/文字)
    }

def get_closest_color_name(rgb: tuple) -> str:
    """基于欧氏距离在高级色彩字典中寻找最接近的颜色名"""
    min_dist = float('inf')
    closest_name = "未知颜色"
    r1, g1, b1 = rgb
    for name, (r2, g2, b2) in COLOR_DICT.items():
        # 简单色彩距离 (可加入亮度权重)
        dist = math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def load_tool_config() -> dict:
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'tool.yml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        knowledge_logger.warning(f'[COLOR_PALETTE] 配置文件失败: {str(e)}')
    return {}

def download_image(image_url: str) -> Optional[np.ndarray]:
    try:
        knowledge_logger.info(f'[COLOR_PALETTE] 下载图片: {image_url}')
        config = load_tool_config()
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        domain = config.get('yuemu', {}).get('cdn_domain', 'static.yuemutuku.com')
        if domain in image_url:
            headers['Referer'] = config.get('yuemu', {}).get('website_url', 'https://www.yuemutuku.com')
            
        response = requests.get(image_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        pil_image = Image.open(BytesIO(response.content))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception as e:
        knowledge_logger.error(f'[COLOR_PALETTE] 下载失败: {str(e)}')
        return None

def extract_color_palette(image: np.ndarray, n_colors: int = 5) -> List[Dict]:
    """在 CIE LAB 感知色彩空间进行聚类，更加符合人类视觉"""
    try:
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        height, width = image_lab.shape[:2]
        
        max_dim = 150
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image_lab = cv2.resize(image_lab, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_NEAREST)
        
        pixels = np.float32(image_lab.reshape(-1, 3))
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_PP_CENTERS
        
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        
        # 将聚类中心转回 RGB 空间
        centers_lab = np.uint8([centers])
        centers_bgr = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2BGR)[0]
        centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB)[0]
        
        unique, counts = np.unique(labels, return_counts=True)
        total_pixels = len(labels)
        
        color_info = []
        for i, (label, count) in enumerate(zip(unique, counts)):
            rgb = tuple(int(x) for x in centers_rgb[label])
            color_info.append({
                "rgb": {"r": rgb[0], "g": rgb[1], "b": rgb[2]},
                "hex": rgb_to_hex(rgb),
                "name": get_closest_color_name(rgb),
                "percentage": round((count / total_pixels) * 100, 2),
                "suggested_text": get_text_color_suggestion(rgb),
                "schemes": get_advanced_color_schemes(rgb)
            })
            
        color_info.sort(key=lambda x: x['percentage'], reverse=True)
        for i, c in enumerate(color_info):
            c['rank'] = i + 1
            
        return color_info
    except Exception as e:
        knowledge_logger.error(f'[COLOR_PALETTE] 聚类异常: {str(e)}')
        return []

def generate_palette_report(colors: List[Dict], image_url: str) -> str:
    if not colors:
        return "抱歉，无法提取图片的色板信息。"
    
    dom = colors[0]
    report = ["**视觉色彩诊断报告**\n\n"]
    
    # 核心分析
    report.append(f"## 画面主导色\n")
    report.append(f"本图视觉面积最大的是 **{dom['name']}** `{dom['hex']}` (占比 {dom['percentage']}%)。")
    report.append(f"如果以此图作为背景卡片，文字建议使用 **{dom['suggested_text']}** 以保证可读性。\n\n")
    
    # 调色板
    report.append("## 核心色卡矩阵\n")
    for c in colors:
        report.append(f"- **{c['rank']}. {c['name']}** `{c['hex']}` (占比 {c['percentage']}%) | RGB({c['rgb']['r']},{c['rgb']['g']},{c['rgb']['b']})\n")
    
    # 专业 UI 衍生搭配推荐 (基于主色)
    schemes = dom['schemes']
    report.append("\n## UI/UX 配色衍生方案\n")
    report.append(f"- **对比色(强调色)**: `{schemes['互补色']}` (用于购买按钮、通知徽标、需要极强视觉冲击的焦点)\n")
    report.append(f"- **邻近色(和谐色)**: `{schemes['同类色_暖']}` / `{schemes['同类色_冷']}` (用于渐变过渡、次要信息块)\n")
    report.append(f"- **浅色背景衍生**: `{schemes['浅色衍生']}` (同色相极浅，适合作为容器背景)\n")
    report.append(f"- **深色文字衍生**: `{schemes['深色衍生']}` (同色相极深，适合作为优雅的标题文本色)\n")
    
    return "".join(report)

@tool(description="""提取图片的主色调色板与高级色彩分析。
功能说明：
1. 使用 CIE LAB 视觉感知空间和 KMeans 聚类提取主色。
2. 专业推导：计算互补色、同类色、三角色及 UI 衍生背景与文本色。
3. 高级命名：自动匹配莫兰迪色、中国传统色等高质感颜色名称。

使用场景：
- 用户想了解图片的主要颜色、提取色板、分析配色。
- 设计师需要提取图片配色方案，求取专业排版、UI色块搭配建议。

参数说明：
- image_url (必需): 图片的 URL 地址
- n_colors (可选): 要提取的颜色数量，默认 5
""")
def extract_image_color_palette(image_url: str, n_colors: Optional[int] = 5) -> str:
    try:
        if not image_url: return "请提供图片 URL"
        if n_colors < 2 or n_colors > 10: n_colors = 5
        
        image = download_image(image_url)
        if image is None: return "抱歉，下载图片失败。"
        
        colors = extract_color_palette(image, n_colors=n_colors)
        if not colors: return "提取色板失败。"
        
        return generate_palette_report(colors, image_url)
        
    except Exception as e:
        knowledge_logger.error(f'[COLOR_PALETTE] 全局失败: {str(e)}')
        return f"色彩分析崩溃: {str(e)}"