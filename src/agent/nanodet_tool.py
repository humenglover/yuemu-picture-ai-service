import requests
import json
import base64
import os
import yaml
from collections import Counter
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger
from .context import get_sa_token

# COCO 80 类的中英文对照字典
COCO_CLASSES_ZH = {
    'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorcycle': '摩托车', 'airplane': '飞机', 
    'bus': '公交车', 'train': '火车', 'truck': '卡车', 'boat': '船', 'traffic light': '红绿灯', 
    'fire hydrant': '消防栓', 'stop sign': '停止标志', 'parking meter': '停车收费表', 'bench': '长椅', 
    'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马', 'sheep': '羊', 'cow': '牛', 'elephant': '大象', 
    'bear': '熊', 'zebra': '斑马', 'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞', 
    'handbag': '手提包', 'tie': '领带', 'suitcase': '手提箱', 'frisbee': '飞盘', 'skis': '滑雪板', 
    'snowboard': '单板滑雪', 'sports ball': '运动球', 'kite': '风筝', 'Kite': '风筝', 'baseball bat': '棒球棒', 
    'baseball glove': '棒球手套', 'skateboard': '滑板', 'surfboard': '冲浪板', 'tennis racket': '网球拍', 
    'bottle': '瓶子', 'wine glass': '高脚杯', 'cup': '杯子', 'fork': '叉子', 'knife': '刀', 'spoon': '勺子', 
    'bowl': '碗', 'banana': '香蕉', 'apple': '苹果', 'sandwich': '三明治', 'orange': '橙子', 
    'broccoli': '西兰花', 'carrot': '胡萝卜', 'hot dog': '热狗', 'pizza': '比萨', 'donut': '甜甜圈', 
    'cake': '蛋糕', 'chair': '椅子', 'couch': '沙发', 'potted plant': '盆栽', 'bed': '床', 
    'dining table': '餐桌', 'toilet': '马桶', 'tv': '电视', 'laptop': '笔记本电脑', 'mouse': '鼠标', 
    'remote': '遥控器', 'keyboard': '键盘', 'cell phone': '手机', 'microwave': '微波炉', 'oven': '烤箱', 
    'toaster': '烤面包机', 'sink': '水槽', 'refrigerator': '冰箱', 'book': '书', 'clock': '时钟', 
    'vase': '花瓶', 'scissors': '剪刀', 'teddy bear': '泰迪熊', 'hair drier': '吹风机', 'toothbrush': '牙刷'
}

def load_tool_config() -> dict:
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'tool.yml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except:
        pass
    return {}

@tool(description="""使用 NanoDet-Plus 识别图片中的具体物体（如人、车、猫、狗等）。
适合需要获取物体种类和精确计数的场景。检测后会自动上传一张带有边框的可视化图片。
注意：不适合用来识别抽象的风景、颜色、画风等，只对具体的独立实体有效。""")
def nanodet_object_detection(image_url: str) -> str:
    """对在线图片进行目标检测，识别图片中的具体物体内容，并将标注结果上传服务器。"""
    try:
        from main import detect_service
        if detect_service is None:
            return "NanoDet服务未初始化"
        
        # 1. 下载图片处理防盗链
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        config = load_tool_config()
        yuemu = config.get('yuemu', {})
        if yuemu.get('cdn_domain', 'static.yuemutuku.com') in image_url:
            headers['Referer'] = yuemu.get('website_url', 'https://www.yuemutuku.com')

        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 2. 调用检测服务（传入字节流）
        result = detect_service.detect(response.content)
        detections = result.get('detections', [])
        
        if not detections:
            return "没有在图中检测到预设的常见物体类别（如人、车、动物等独立实体）。"
            
        # 3. 中文化并进行数量聚合统计
        translated_labels = [COCO_CLASSES_ZH.get(d['label'].lower(), d['label']) for d in detections]
        counter = Counter(translated_labels)
        
        report = ["**NanoDet-Plus 目标检测分析结果**\n"]
        for item, count in counter.items():
            report.append(f"- **{item}**: 发现 {count} 个")
        report.append(f"\n*共计识别出 {len(detections)} 个实体。*")

        # 4. 上传带有识别边框的图像到 Java 后端
        annotated_b64 = result.get('annotatedImageBase64')
        annotated_url = None
        
        if annotated_b64:
            token = get_sa_token()
            if token:
                try:
                    img_bytes = base64.b64decode(annotated_b64)
                    
                    try:
                        from model.factory import load_config
                        rag_config = load_config()
                        java_base_url = rag_config.get("java_backend_url", "http://127.0.0.1:8123/api")
                    except Exception:
                        java_base_url = "http://127.0.0.1:8123/api"
                        
                    upload_url = f"{java_base_url}/picture/upload/postimage"
                    headers_up = {"satoken": token}
                    data_up = {
                        "picName": "NanoDet智能识别结果",
                        "introduction": "由 NanoDet-Plus 模型检测并标注出的结果图片"
                    }
                    
                    # 写入临时文件，榨取其天然宽高及主色调，同时保障无文件防泄露
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    try:
                        with open(tmp_path, 'wb') as tmp_f:
                            tmp_f.write(img_bytes)
                        from utils.image_utils import analyze_local_image_attributes
                        attrs = analyze_local_image_attributes(tmp_path)
                        if attrs:
                            data_up['picWidth'] = attrs['picWidth']
                            data_up['picHeight'] = attrs['picHeight']
                            data_up['picScale'] = attrs['picScale']
                            data_up['picColor'] = attrs['picColor']
                    finally:
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except:
                                pass
                                
                    files_up = {'file': ('nanodet_annotated.jpg', img_bytes, 'image/jpeg')}
                    
                    knowledge_logger.info(f"[NANODET_TOOL] 正在上传标注图像...")
                    up_resp = requests.post(upload_url, headers=headers_up, data=data_up, files=files_up, timeout=30)
                    if up_resp.status_code == 200 and up_resp.json().get("code") == 0:
                        annotated_url = up_resp.json().get("data", {}).get("url")
                        knowledge_logger.info(f"[NANODET_TOOL] 标注图像上传成功: {annotated_url}")
                except Exception as up_e:
                    knowledge_logger.error(f"[NANODET_TOOL_ERROR] 上传标注图像失败: {str(up_e)}")
            else:
                knowledge_logger.warning("[NANODET_TOOL] 无身份凭证，跳过图像上传。")

        if annotated_url:
            report.append(f"\n[附图: {annotated_url}]")
            report.append("\n> *我已经将包含目标检测框的图片附在上方。请确保你的最终回答里完整保留了这个 [附图: URL] 标记！*")

        return "\n".join(report)
        
    except Exception as e:
        error_msg = f"NanoDet检测失败: {str(e)}"
        knowledge_logger.error(f'[NANODET_TOOL_ERROR] {error_msg}')
        return f"检测工具执行异常: {error_msg}"
