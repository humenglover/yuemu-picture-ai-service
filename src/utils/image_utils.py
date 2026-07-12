#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共图片分析与属性处理工具类 (Image Attributes Analyzer)
"""

import os
from PIL import Image
import cv2
import numpy as np
from utils.log_utils import knowledge_logger

def analyze_local_image_attributes(image_path: str) -> dict:
    """提取本地图片的宽高、比例及主色调(picColor)的十六进制形式"""
    try:
        if not image_path or not os.path.exists(image_path):
            return {}
            
        # 1. 宽高和比例分析
        with Image.open(image_path) as img:
            width, height = img.size
            scale = round(width / height, 2)
            
        # 2. KMeans (K=1) 色彩榨取算法提取核心主视觉色调
        img_cv = cv2.imread(image_path)
        if img_cv is not None:
            # 缩放至 50x50 以极大提速，降低像素量而不丢失核心色相分量
            img_small = cv2.resize(img_cv, (50, 50), interpolation=cv2.INTER_NEAREST)
            img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            pixels = img_small.reshape(-1, 3)
            pixels = np.float32(pixels)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            dom_rgb = tuple(int(x) for x in centers[0])
            hex_color = "#{:02x}{:02x}{:02x}".format(dom_rgb[0], dom_rgb[1], dom_rgb[2])
        else:
            hex_color = "#FFFFFF"
            
        return {
            "picWidth": width,
            "picHeight": height,
            "picScale": scale,
            "picColor": hex_color
        }
    except Exception as e:
        knowledge_logger.error(f'[IMAGE_UTILS] 分析本地图片属性异常: {str(e)}')
        return {}
