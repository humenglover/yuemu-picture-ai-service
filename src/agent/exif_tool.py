#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片元数据提取工具 - 基于 Pillow + ExifRead 深度解析
能够提取极尽详细的相机参数、镜头型号、拍摄时间、设备型号及定位信息
并且即使图片没有 EXIF，也能返回格式、尺寸等基础属性。
"""

import requests
from io import BytesIO
from PIL import Image
import exifread
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger
import yaml
import os

def load_tool_config() -> dict:
    """加载工具配置文件"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'tool.yml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        knowledge_logger.warning(f'[EXIF_TOOL] 加载配置文件失败: {str(e)}')
    return {}

def _convert_ratio(val):
    """转换 ExifRead 的 Ratio 对象为易读格式"""
    if hasattr(val, 'num') and hasattr(val, 'den'):
        if val.den == 0:
            return str(val.num)
        # 快门速度习惯保留分数，光圈焦距习惯保留小数
        return f"{val.num}/{val.den}"
    return str(val)

def _convert_gps(coords, ref):
    """转换 GPS 坐标为十进制度数"""
    try:
        d = float(coords[0].num) / float(coords[0].den)
        m = float(coords[1].num) / float(coords[1].den)
        s = float(coords[2].num) / float(coords[2].den)
        decimal = d + (m / 60.0) + (s / 3600.0)
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal
    except Exception:
        return None

@tool(description="""提取图片的基础属性与详细的 EXIF 元数据（包括相机参数与拍摄信息）。
功能说明：
1. 始终提取：图片格式、色彩模式、原始分辨率大小、文件体积
2. 深度提取(若有)：拍摄设备品牌/型号、镜头型号、焦距(含等效焦距)、光圈、快门、ISO
3. 提取拍摄时间、后期软件信息
4. 提取 GPS 经纬度信息

输入参数:
- image_url (必需): 图片的 URL 地址

返回格式:
返回结构化的文本报告，哪怕图片因为被压缩丢失了相机参数，也会提供格式与尺寸大小。""")
def extract_image_exif(image_url: str) -> str:
    """提取并解析图片的 EXIF 数据"""
    try:
        knowledge_logger.info(f'[EXIF_TOOL] 开始提取深度 EXIF | URL: {image_url}')
        
        if not image_url:
            return "请提供有效的图片 URL。"
        
        # 1. 组装请求头，绕过防盗链
        config = load_tool_config()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        yuemu_config = config.get('yuemu', {})
        cdn_domain = yuemu_config.get('cdn_domain', 'static.yuemutuku.com')
        website_url = yuemu_config.get('website_url', 'https://www.yuemutuku.com')
        
        if cdn_domain in image_url:
            headers['Referer'] = website_url
            
        # 2. 下载图片二进制数据
        response = requests.get(image_url, headers=headers, timeout=15)
        response.raise_for_status()
        img_bytes = response.content
        file_size_kb = len(img_bytes) / 1024
        
        report_parts = ["**图片属性与元数据报告**\n\n"]
        
        # 3. Pillow 解析基础属性 (保底能力)
        try:
            image = Image.open(BytesIO(img_bytes))
            report_parts.append(f"**基础属性**\n")
            report_parts.append(f"- **格式**: {image.format}\n")
            report_parts.append(f"- **尺寸**: {image.width} × {image.height} 像素\n")
            report_parts.append(f"- **色彩模式**: {image.mode}\n")
            report_parts.append(f"- **文件大小**: {file_size_kb:.1f} KB\n\n")
        except Exception as e:
            knowledge_logger.error(f'[EXIF_TOOL] Pillow 解析失败: {str(e)}')
            return f"无法读取该图像文件，可能文件已损坏或格式不支持 ({str(e)})。"
            
        # 4. ExifRead 深度提取相机数据
        tags = exifread.process_file(BytesIO(img_bytes), details=False)
        
        if not tags:
            report_parts.append("> **说明**: 该图片未包含任何相机 EXIF 元数据。可能是由于网络传输、平台压缩或后期处理时擦除了摄影信息。")
            return "".join(report_parts)
            
        # -- 提取设备信息 --
        make = str(tags.get('Image Make', ''))
        model = str(tags.get('Image Model', ''))
        lens = str(tags.get('EXIF LensModel', ''))
        software = str(tags.get('Image Software', ''))
        
        if make or model:
            report_parts.append("**拍摄设备**\n")
            if make and make not in model:
                report_parts.append(f"- **相机**: {make} {model}\n")
            else:
                report_parts.append(f"- **相机**: {model or make}\n")
            if lens:
                report_parts.append(f"- **镜头**: {lens}\n")
            if software:
                report_parts.append(f"- **后期/系统**: {software}\n")
            report_parts.append("\n")
            
        # -- 提取时间信息 --
        dt = tags.get('EXIF DateTimeOriginal', tags.get('Image DateTime', ''))
        if dt:
            report_parts.append(f"**拍摄时间**: {str(dt)}\n\n")
            
        # -- 提取曝光参数 --
        iso = str(tags.get('EXIF ISOSpeedRatings', '未知'))
        focal = tags.get('EXIF FocalLength')
        focal_35 = tags.get('EXIF FocalLengthIn35mmFilm')
        f_num = tags.get('EXIF FNumber')
        exposure = tags.get('EXIF ExposureTime')
        
        if any(x is not None for x in [focal, f_num, exposure]) or iso != '未知':
            report_parts.append("**曝光参数**\n")
            
            # 处理焦距
            if focal:
                val = _convert_ratio(focal.values[0] if hasattr(focal, 'values') else focal)
                try: 
                    f_val = float(eval(val)) if '/' in val else float(val)
                    focal_str = f"{f_val:.1f}mm"
                except: focal_str = val
                
                if focal_35:
                    report_parts.append(f"- **焦距**: {focal_str} (等效35mm: {str(focal_35)}mm)\n")
                else:
                    report_parts.append(f"- **焦距**: {focal_str}\n")
                    
            # 处理光圈
            if f_num:
                val = _convert_ratio(f_num.values[0] if hasattr(f_num, 'values') else f_num)
                try: 
                    fv = float(eval(val)) if '/' in val else float(val)
                    report_parts.append(f"- **光圈**: f/{fv:.1f}\n")
                except: report_parts.append(f"- **光圈**: f/{val}\n")
                
            # 处理快门
            if exposure:
                val = _convert_ratio(exposure.values[0] if hasattr(exposure, 'values') else exposure)
                report_parts.append(f"- **快门**: {val}s\n")
                
            if iso != '未知':
                report_parts.append(f"- **ISO**: {iso}\n")
            report_parts.append("\n")
            
        # -- 提取 GPS --
        gps_lat = tags.get('GPS GPSLatitude')
        gps_lat_ref = tags.get('GPS GPSLatitudeRef')
        gps_lon = tags.get('GPS GPSLongitude')
        gps_lon_ref = tags.get('GPS GPSLongitudeRef')
        
        if gps_lat and gps_lon and gps_lat_ref and gps_lon_ref:
            lat_dec = _convert_gps(gps_lat.values, str(gps_lat_ref.values))
            lon_dec = _convert_gps(gps_lon.values, str(gps_lon_ref.values))
            if lat_dec is not None and lon_dec is not None:
                report_parts.append(f"**定位信息**: 包含 GPS 数据 (纬度 {lat_dec:.5f}, 经度 {lon_dec:.5f})\n")

        return "".join(report_parts)
        
    except Exception as e:
        error_msg = f"提取 EXIF 时发生错误: {str(e)}"
        knowledge_logger.error(f'[EXIF_TOOL] {error_msg}')
        import traceback
        knowledge_logger.error(f'[EXIF_TOOL] 堆栈: {traceback.format_exc()}')
        return "提取图片元数据失败，发生意外异常。"
