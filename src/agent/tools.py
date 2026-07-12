from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger

class ToolRegistry:
    """工具注册表，用于管理和加载工具"""
    def __init__(self):
        self._tools = []

    def register(self, tool_func):
        """注册一个工具"""
        self._tools.append(tool_func)
        return tool_func

    @property
    def tools(self):
        """返回所有已注册的工具列表"""
        return self._tools

# 创建全局注册表
registry = ToolRegistry()

# --- 导入并注册业务工具 ---
try:
    from .biz_tool import (
        upload_picture_to_space, list_available_spaces, delete_picture, 
        search_pictures, search_pictures_by_image, search_site, get_my_personal_data,
        get_picture_detail_data, get_post_detail_data, 
        get_user_detail_data, get_follow_or_fan_list_data
    )
    registry.register(upload_picture_to_space)
    registry.register(list_available_spaces)
    registry.register(delete_picture)
    registry.register(search_pictures)
    registry.register(search_pictures_by_image)
    registry.register(search_site)
    registry.register(get_my_personal_data)
    registry.register(get_picture_detail_data)
    registry.register(get_post_detail_data)
    registry.register(get_user_detail_data)
    registry.register(get_follow_or_fan_list_data)
except ImportError:
    pass

# --- 导入并注册 RAG 工具 ---
try:
    from .rag_tool import rag_summarize
    registry.register(rag_summarize)
except ImportError:
    pass

# --- 导入并注册 NanoDet 目标检测工具 ---
try:
    from .nanodet_tool import nanodet_object_detection
    registry.register(nanodet_object_detection)
except ImportError:
    pass

# --- 导入并注册通用/外部工具 ---
try:
    from .tavily_tool import tavily_search_tool
    registry.register(tavily_search_tool)
except ImportError:
    pass

try:
    from .datetime_tool import datetime_tool
    registry.register(datetime_tool)
except ImportError:
    pass

# --- 导入并注册 TTS 工具 ---
try:
    from .tts_tool import tts_reply
    registry.register(tts_reply)
except ImportError:
    pass

# --- 导入并注册天气查询工具 ---
try:
    from .weather_tool import weather_query_tool
    registry.register(weather_query_tool)
except ImportError:
    pass

# --- 导入并注册 Pexels 图片搜索工具 ---
try:
    from .pexels_tool import search_pexels_images, get_pexels_curated
    registry.register(search_pexels_images)
    registry.register(get_pexels_curated)
except ImportError:
    pass

# --- 导入并注册系统监控工具 ---
try:
    from .system_monitor_tool import check_system_status, get_top_processes
    registry.register(check_system_status)
    registry.register(get_top_processes)
except ImportError:
    pass

# --- 导入并注册色板提取工具 ---
try:
    from .color_palette_tool import extract_image_color_palette
    registry.register(extract_image_color_palette)
except ImportError:
    pass

# --- 导入并注册 EXIF 元数据提取工具 ---
try:
    from .exif_tool import extract_image_exif
    registry.register(extract_image_exif)
except ImportError:
    pass

# --- 导入并注册图片上传工具 ---
try:
    from .image_upload_tool import upload_local_image
    registry.register(upload_local_image)
except ImportError:
    pass

# --- 导入并注册智能裁剪工具 ---
try:
    from .smart_crop_tool import smart_crop_and_upload
    registry.register(smart_crop_and_upload)
except ImportError:
    pass

# --- 导入并注册画质增强工具 ---
try:
    from .image_enhance_tool import enhance_and_upload
    registry.register(enhance_and_upload)
except ImportError:
    pass

# --- 导入并注册水印工具 ---
try:
    from .watermark_tool import add_watermark_and_upload
    registry.register(add_watermark_and_upload)
except ImportError:
    pass

# --- 导入并注册艺术色彩滤镜工具 ---
try:
    from .filter_tool import apply_filter_and_upload
    registry.register(apply_filter_and_upload)
except ImportError:
    pass

# --- 导入并注册艺术装裱工具 ---
try:
    from .frame_tool import apply_frame_and_upload
    registry.register(apply_frame_and_upload)
except ImportError:
    pass

# --- 导入并注册智能艺术拼接卡片工具 ---
try:
    from .card_generator_tool import generate_art_card_and_upload
    registry.register(generate_art_card_and_upload)
except ImportError:
    pass

# --- 导入并注册 AI 文生图生成工具 ---
try:
    from .image_generation_tool import generate_image
    registry.register(generate_image)
except ImportError:
    pass

# --- 导入并注册 AI 封面生成工具 ---
try:
    from .ai_cover_generator_tool import generate_ai_cover
    registry.register(generate_ai_cover)
except ImportError:
    pass

# --- 导入并注册核心逻辑兜底工具 ---
try:
    from .core_tools import think, noop, get_session_history, search_long_term_memory
    registry.register(think)
    registry.register(noop)
    registry.register(get_session_history)
    registry.register(search_long_term_memory)
except ImportError:
    pass

# 向后兼容
available_tools = registry.tools