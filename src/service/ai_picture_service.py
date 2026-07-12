import json
import asyncio
from typing import AsyncGenerator
from utils.log_utils import app_logger
from model.factory import create_chat_model
from agent.context import set_sa_token
from langchain_core.messages import HumanMessage, SystemMessage

class AIPictureService:
    def __init__(self):
        pass

    async def process_ai_image_description_stream(self, request) -> AsyncGenerator[str, None]:
        """流式生成图片标题和简介（识图配文）"""
        if hasattr(request, 'sa_token') and request.sa_token:
            set_sa_token(request.sa_token)
            
        yield f"event: status\ndata: {json.dumps({'status': '正在观察图片细节...'})}\n\n"
        await asyncio.sleep(0.1)
        
        system_prompt = (
            "你是一个专业的图片分享平台编辑。请仔细观察用户提供的图片，写一个标题和一段简介。\n"
            "要求：\n"
            "1. 第一行必须是以 #TITLE# 开头的标题，例如：#TITLE# 标题内容（限制在15字以内，精炼且吸引人）。\n"
            "2. 第二行及以后是正文简介，要求100字以内，客观且生动地描述画面内容，可以适当带1-2个Emoji。\n"
            "注意：严禁出现任何多余的开场白或结束语，严格遵循格式。"
        )
        
        human_content = [
            {"type": "text", "text": "请帮我为这张图片生成标题和简介。"},
            {"type": "image_url", "image_url": {"url": request.image_url}}
        ]
        
        chat_model = create_chat_model()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]

        try:
            async for chunk in chat_model.astream(messages):
                if chunk.content:
                    yield f"event: content_chunk\ndata: {json.dumps({'text': chunk.content})}\n\n"
        except Exception as e:
            app_logger.error(f"图片识别配文异常: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        
        yield f"event: done\ndata: {json.dumps({'status': '生成完毕'})}\n\n"

ai_picture_service = AIPictureService()
