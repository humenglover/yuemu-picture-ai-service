import os
import random
import json
import uuid
import tempfile
import requests
import cv2
from typing import Optional
from langchain_core.tools import tool
from PIL import Image, ImageDraw
from utils.log_utils import knowledge_logger

from agent.pexels_tool import search_pexels_images
from agent.color_palette_tool import download_image, extract_color_palette
from agent.card_generator_tool import get_elegant_font, wrap_text_by_width, hex_to_rgb

@tool(description="""生成带排版的智能封面图工具。
当用户需要生成文章封面图、配图封面等时调用此工具。它会从 Pexels 获取相关高质量配图，并进行美学排版。
参数：
- title (必须): 主标题文字，尽量精炼，作为封面的核心视觉。
- sub_title (可选): 副标题或辅助说明文字。
- style_id (可选): 封面风格。
返回：
包含生成图片的上传结果 JSON（内含 url 字段）。在回答时务必使用标准的 [附图: url] 格式呈现。
""")
def generate_ai_cover(title: str, sub_title: Optional[str] = "", style_id: Optional[int] = None) -> str:
    """生成带排版的智能封面图并上传返回结果。"""
    try:
        # 1. 随机选择本地背景图
        # python-rag的资源路径：代码/python-rag/src/assets/frames
        frames_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/frames"))
        if os.path.exists(frames_dir):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
            frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(valid_extensions)]
        else:
            frame_files = []
            
        if not frame_files:
            # 兜底图：如果本地没图，尝试用网络图（治愈系风景）
            img_url = "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&cs=tinysrgb&h=650&w=940"
            pil_bg = download_image(img_url)
            if pil_bg is None:
                return json.dumps({"error": "未找到预设的封面背景图"}, ensure_ascii=False)
            pil_bg = Image.fromarray(cv2.cvtColor(pil_bg, cv2.COLOR_BGR2RGB))
        else:
            import secrets
            frame_path = os.path.join(frames_dir, secrets.choice(frame_files))
            pil_bg = Image.open(frame_path).convert("RGB")
        
        # 2. 读取并处理背景图
        orig_w, orig_h = pil_bg.size
        
        # 小红书标准封面比例 3:4，设置 900x1200
        card_w, card_h = 900, 1200
        
        # 计算缩放比例并居中裁剪以填满 900x1200 (类似 object-fit: cover)
        ratio_w = card_w / orig_w
        ratio_h = card_h / orig_h
        ratio = max(ratio_w, ratio_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        
        pil_bg = pil_bg.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 居中裁剪
        left = (new_w - card_w) // 2
        top = (new_h - card_h) // 2
        card = pil_bg.crop((left, top, left + card_w, top + card_h))
        
        # 3. 添加遮罩层 (15% 黑色) 保证文字可读性，降低暗度
        overlay = Image.new('RGBA', (card_w, card_h), (0, 0, 0, int(255 * 0.15)))
        card = Image.alpha_composite(card.convert('RGBA'), overlay).convert('RGB')
        
        draw = ImageDraw.Draw(card)
        
        # 4. 居中排版文本
        font_title = get_elegant_font("bold", size=72)
        font_sub = get_elegant_font("regular", size=36)
        
        # 估算总文字高度以居中
        line_h = 100
        title_lines = wrap_text_by_width(title, font_title, max_width=card_w - 120)
        total_text_h = len(title_lines) * line_h
        
        sub_lines = []
        if sub_title and sub_title != "默认分类":
            sub_lines = wrap_text_by_width(sub_title, font_sub, max_width=card_w - 120)
            total_text_h += 60 + len(sub_lines) * 60
            
        current_y = (card_h - total_text_h) // 2
        
        def draw_centered_text(draw, text, font, y_pos, fill):
            try:
                bbox = font.getbbox(text)
                w = bbox[2] - bbox[0]
            except Exception:
                w = font.getsize(text)[0]
            x_pos = (card_w - w) // 2
            draw.text((x_pos, y_pos), text, fill=fill, font=font)
        
        text_rgb = (255, 255, 255)
        
        # 绘制主标题
        for line in title_lines:
            draw_centered_text(draw, line, font_title, current_y, text_rgb)
            current_y += line_h
            
        # 绘制副标题（分类）
        if sub_lines:
            current_y += 20
            # 绘制小横线修饰
            line_w = 60
            draw.line([(card_w - line_w) // 2, current_y - 20, (card_w + line_w) // 2, current_y - 20], fill=(255, 255, 255), width=4)
            current_y += 30
            
            sub_color = (230, 230, 230)
            for line in sub_lines:
                draw_centered_text(draw, line, font_sub, current_y, sub_color)
                current_y += 60
        
        # 保存并上传
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"ai_cover_{uuid.uuid4().hex}.jpg")
        card.save(temp_file, format="JPEG", quality=95)
        
        from .context import get_sa_token
        token = get_sa_token()
        if not token:
            return json.dumps({"error": "未提供身份凭证，无法上传封面。"}, ensure_ascii=False)
            
        try:
            from model.factory import load_config
            java_base_url = load_config().get("java_backend_url", "http://127.0.0.1:8123/api")
        except Exception:
            java_base_url = "http://127.0.0.1:8123/api"
            
        upload_url = f"{java_base_url}/picture/upload/postimage"
        upload_headers = {"satoken": token}
        params = {
            'spaceId': -1,
            'picName': title[:20] if title else "AI封面"
        }
        
        from utils.image_utils import analyze_local_image_attributes
        attrs = analyze_local_image_attributes(temp_file)
        if attrs:
            params['picWidth'] = attrs['picWidth']
            params['picHeight'] = attrs['picHeight']
            params['picScale'] = attrs['picScale']
            params['picColor'] = attrs['picColor']
            
        with open(temp_file, 'rb') as f:
            files = {'file': ('ai_cover.jpg', f, 'image/jpeg')}
            upload_response = requests.post(upload_url, headers=upload_headers, params=params, files=files, timeout=60)
            upload_result = upload_response.json()
            
        try:
            os.remove(temp_file)
        except Exception:
            pass
            
        if upload_response.status_code == 200 and upload_result.get("code") == 0:
            data = upload_result.get("data", {})
            result_data = {
                "type": "ai_cover_generated",
                "msg": f"AI封面生成成功: {title}",
                "url": data.get("url"),
                "thumbnailUrl": data.get("thumbnailUrl"),
                "name": data.get("name"),
                "picWidth": data.get("picWidth"),
                "picHeight": data.get("picHeight"),
                "picScale": data.get("picScale"),
                "picColor": data.get("picColor")
            }
            return json.dumps(result_data, ensure_ascii=False)
        else:
            return json.dumps({"error": f"上传封面图失败: {upload_result.get('message', '未知')}"}, ensure_ascii=False)
            
    except Exception as e:
        knowledge_logger.error(f"[AI_COVER_GENERATOR] 生成封面异常: {str(e)}")
        import traceback
        knowledge_logger.error(traceback.format_exc())
        return json.dumps({"error": f"生成封面失败: {str(e)}"}, ensure_ascii=False)
