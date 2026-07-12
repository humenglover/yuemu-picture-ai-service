from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile, Response
from typing import List, Optional, Dict, Any
import sys
import os

from datetime import datetime
import json
import time
import hashlib
import asyncio
import shutil
from concurrent.futures import ThreadPoolExecutor

# 添加项目根路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

# 配置强制输出（保留配置但不再使用）
os.environ["FORCE_CONSOLE_OUTPUT"] = "True"

from model.factory import load_config, create_chat_model

# 加载配置
config = load_config()
concurrency_config = config.get('concurrency', {})
thread_pool_config = concurrency_config.get('thread_pool', {})
max_workers = thread_pool_config.get('max_workers', 20)
thread_pool = ThreadPoolExecutor(max_workers=max_workers)

# 导入工具类和模型
from utils.log_utils import app_logger, upload_logger, knowledge_logger
from utils.file_utils import log_event, calculate_md5, check_md5_exists_only, check_md5_exists_permanently, \
    save_md5_with_filename, save_md5_permanently, find_filename_by_md5, get_file_documents

from model.dto.rag_request_dto import RAGRequestDTO, StreamRAGRequestDTO, AIPostRequestDTO, AIPictureRequestDTO, PureLLMChatRequest
from model.dto.rag_response_dto import RAGResponseDTO
from model.common.response_wrapper import ResponseWrapper
from model.common.constants import RAGConstants, HttpStatusCodes

from rag.vector_store import VectorStoreManager
from rag.rag_summarize import RAGSummarizer
from service.knowledge_management_service import knowledge_management_service
from service.rag_service import RAGService
from agent.react_agent import ReactAgent
from service.nanodet_service import NanoDetService
from service.image_service import ImageService
from agent.context import set_sa_token
from service.comment_moderation_service import comment_moderation_service

# 全局组件
vector_store_manager = None
rag_summarizer = None
agent = None
rag_service = None
detect_service = None
image_service = None



def init_core_components():
    """初始化核心组件"""
    global vector_store_manager, rag_summarizer, agent, rag_service, detect_service, image_service

    app_logger.info("\n" + "=" * 80)
    app_logger.info(f"初始化RAG服务 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    app_logger.info("=" * 80)

    try:
        # 初始化向量存储
        app_logger.info("初始化向量存储管理器...")
        vector_store_manager = VectorStoreManager()

        # 初始化RAG摘要器
        app_logger.info("初始化RAG摘要器...")
        rag_summarizer = RAGSummarizer(vector_store_manager)

        # 初始化Agent
        app_logger.info("初始化ReactAgent...")
        agent = ReactAgent()
        rag_service = RAGService(rag_summarizer, agent)

        app_logger.info("RAG组件和Agent初始化成功！")
        app_logger.info("=" * 80 + "\n")

    except Exception as agent_err:
        app_logger.warning(f"Agent初始化失败: {str(agent_err)}，将仅启用RAG功能")
        try:
            vector_store_manager = VectorStoreManager()
            rag_summarizer = RAGSummarizer(vector_store_manager)
            rag_service = RAGService(rag_summarizer)

            app_logger.info("RAG组件初始化成功！")
            app_logger.info("=" * 80 + "\n")
        except Exception as rag_err:
            app_logger.error(f"RAG组件初始化失败: {str(rag_err)}")
            sys.exit(1)

    # ============================================================
    # NanoDet-Plus 目标检测服务 - 始终加载（模型约3.7MB，远小于 YOLOv8）
    # ============================================================
    try:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        yolo_model_path = os.path.join(models_dir, "nanodet-plus-m_320.onnx")
        app_logger.info(f"初始化NanoDet-Plus服务 | 模型路径: {yolo_model_path}")
        detect_service = NanoDetService(yolo_model_path)
        app_logger.info("NanoDet-Plus服务初始化成功！")
    except Exception as yolo_err:
        app_logger.error(f"NanoDet-Plus服务初始化失败: {str(yolo_err)}")
        detect_service = None

    # ============================================================
    # 图像处理服务 (去背景、人脸打码)
    # ============================================================
    try:
        app_logger.info("初始化图像处理服务 (NanoDet + GrabCut 智能抠图)...")
        image_service = ImageService(detect_service=detect_service)
        app_logger.info("图像处理服务初始化成功！")
    except Exception as img_err:
        app_logger.error(f"图像处理服务初始化失败: {str(img_err)}")
        image_service = None

    # ============================================================
    # 文本审核服务 (纯 DFA 版本) 独立初始化
    # 完全解耦，不影响 RAG / 图像服务链路
    # ============================================================
    try:
        app_logger.info("初始化文本审核服务 (纯 DFA 版本)...")
        comment_moderation_service.init()
        app_logger.info("文本审核服务初始化成功！")
    except Exception as cmt_err:
        app_logger.error(f"文本审核服务初始化失败: {str(cmt_err)}，审核接口将返回 503")


# 执行初始化
init_core_components()

# 创建FastAPI应用
app = FastAPI(title="Python RAG Service", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== API接口 ==========
@app.post("/api/rag/sync", response_model=RAGResponseDTO)
async def rag_sync(request: RAGRequestDTO):
    try:
        # 设置鉴权上下文
        set_sa_token(request.sa_token)
        app_logger.info(f"\n处理RAG同步请求 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(thread_pool, rag_service.process_rag_sync, request)
        return result
    except Exception as e:
        error_msg = f"RAG同步处理失败：{str(e)}"
        app_logger.error(f"{error_msg}")
        app_logger.error(f"[RAG_SYNC_ERROR] {error_msg}")
        return ResponseWrapper.error(msg=error_msg)

@app.post("/api/ai/pure-chat", response_model=RAGResponseDTO)
async def pure_llm_chat(request: PureLLMChatRequest):
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from model.vo.rag_vo import RAGResultVO
        
        # 加载基础配置
        config = load_config()
        model_name = request.model if request.model else config.get('chat_model_name', 'qwen3.5-flash')
        temperature = request.temperature if request.temperature is not None else config.get('qwen_temperature', 0.1)
        max_tokens = request.max_tokens if request.max_tokens is not None else config.get('qwen_max_tokens', 2048)
        
        app_logger.info(f"\n[PURE_LLM] 纯AI请求开始 | model: {model_name} | temp: {temperature}")
        
        # 实例化 ChatOpenAI，彻底绕过 Agent/RAG 框架与 Qdrant 检索
        chat_model = ChatOpenAI(
            model=model_name,
            openai_api_key=config['qwen_api_key'],
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={'enable_thinking': False}
        )
        
        # 单次纯调用模型
        raw_response = await chat_model.ainvoke([HumanMessage(content=request.prompt)])
        answer = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
        
        result_vo = RAGResultVO(
            answer=answer.strip(),
            session_id=f"pure_{int(time.time())}",
            top_k=0,
            temperature=temperature,
            total_tokens=0
        )
        
        app_logger.info(f"[PURE_LLM] 纯AI请求成功 | 长度: {len(result_vo.answer)}")
        return ResponseWrapper.success(data=result_vo.model_dump())
    except Exception as e:
        error_msg = f"纯AI接口调用失败：{str(e)}"
        app_logger.error(f"[PURE_LLM_ERROR] {error_msg}")
        return ResponseWrapper.error(msg=error_msg)

@app.post("/api/rag/stream")
async def rag_stream(request: StreamRAGRequestDTO):
    try:
        # 设置鉴权上下文
        set_sa_token(request.sa_token)
        app_logger.info(f"\n处理RAG流式请求 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        from fastapi.responses import StreamingResponse
        return StreamingResponse(rag_service.process_rag_stream(request), media_type="text/event-stream")
    except Exception as e:
        error_msg = f"RAG流式处理失败：{str(e)}"
        app_logger.error(f"{error_msg}")
        app_logger.error(f"[RAG_STREAM_ERROR] {error_msg}")
        return ResponseWrapper.error(msg=error_msg)

@app.post("/api/ai_post/stream")
async def ai_post_stream(request: AIPostRequestDTO):
    try:
        from service.ai_post_service import ai_post_service
        set_sa_token(request.sa_token)
        app_logger.info(f"\n处理AI一键成帖流式请求 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        from fastapi.responses import StreamingResponse
        return StreamingResponse(ai_post_service.process_ai_post_stream(request), media_type="text/event-stream")
    except Exception as e:
        error_msg = f"AI一键成帖处理失败：{str(e)}"
        app_logger.error(f"{error_msg}")
        return ResponseWrapper.error(msg=error_msg)

@app.post("/api/ai_picture/stream")
async def ai_picture_stream(request: AIPictureRequestDTO):
    try:
        from service.ai_picture_service import ai_picture_service
        set_sa_token(request.sa_token)
        app_logger.info(f"\n处理AI识图配文流式请求 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        from fastapi.responses import StreamingResponse
        return StreamingResponse(ai_picture_service.process_ai_image_description_stream(request), media_type="text/event-stream")
    except Exception as e:
        error_msg = f"AI识图配文处理失败：{str(e)}"
        app_logger.error(f"{error_msg}")
        return ResponseWrapper.error(msg=error_msg)

@app.post("/api/ai/image-keywords")
async def get_image_keywords(request: AIPictureRequestDTO):
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # 加载基础配置，使用默认的 qwen3.5-flash 模型
        config = load_config()
        model_name = config.get('chat_model_name', 'qwen3.5-flash')
        
        app_logger.info(f"\n[IMAGE_KEYWORDS] 开始提取图片关键词 | model: {model_name}")
        
        chat_model = create_chat_model(model_name)
        
        system_prompt = (
            "你是一个图片检索助手。请认真观察这张图片，提取出该图片最核心、最具代表性的5-8个简体中文检索关键词。\n"
            "关键词应当包括：图片主体、颜色风格、艺术流派、构图或核心场景。\n"
            "请直接输出这几个关键词，用空格分隔，严禁输出任何多余的解释、Markdown标记、前导词或结束语。"
        )
        
        human_content = [
            {"type": "text", "text": "请提取该图片的检索关键词。"},
            {"type": "image_url", "image_url": {"url": request.image_url}}
        ]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]
        
        raw_response = await chat_model.ainvoke(messages)
        keywords = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
        
        app_logger.info(f"[IMAGE_KEYWORDS] 提取成功 | 结果: {keywords}")
        return ResponseWrapper.success(data={"keywords": keywords.strip()})
    except Exception as e:
        error_msg = f"提取图片关键词失败：{str(e)}"
        app_logger.error(f"[IMAGE_KEYWORDS_ERROR] {error_msg}")
        return ResponseWrapper.error(msg=error_msg)

@app.post("/api/rag/summarize", response_model=RAGResponseDTO)
async def rag_summarize(request: RAGRequestDTO):
    try:
        app_logger.info(f"\n处理RAG摘要请求 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = await rag_service.process_summarize_async(request)
        return result
    except Exception as e:
        error_msg = f"RAG摘要处理失败：{str(e)}"
        app_logger.error(f"{error_msg}")
        return ResponseWrapper.error(msg=error_msg)

@app.get("/api/tts")
async def generate_tts(text: str, voice_type: str = "female_gentle"):
    try:
        app_logger.info(f"处理TTS语音合成请求 | text_length={len(text)}, voice_type={voice_type}")
        from agent.tts_tool import _generate_tencent_tts, VOICE_MAP
        voice_id = VOICE_MAP.get(voice_type, 101001)
        audio_bytes = _generate_tencent_tts(text, voice_id)
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        from fastapi.responses import Response
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        app_logger.error(f"TTS接口失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/upload")
async def upload_knowledge_file(file: UploadFile = File(...)):
    """文件上传接口"""
    response = await knowledge_management_service.process_upload_file(file, vector_store_manager)
    # 根据响应code设置HTTP状态码
    from starlette.responses import JSONResponse
    return JSONResponse(
        content=response.model_dump(),
        status_code=response.code
    )

@app.get("/api/knowledge/list")
async def get_all_knowledge_files(file_type: str = None):
    """获取文件列表接口"""
    return await knowledge_management_service.get_all_knowledge_files_api(file_type)

@app.post("/api/knowledge/clear-all")
async def clear_all_knowledge():
    """清空所有知识库"""
    return await knowledge_management_service.clear_all_knowledge_api(vector_store_manager)

@app.post("/api/knowledge/delete")
async def delete_knowledge_files_by_md5(request: dict):
    """删除文件接口"""
    return await knowledge_management_service.delete_multiple_knowledge_files_by_md5_api(request, vector_store_manager)

@app.get("/api/vector/verify/{file_md5}")
async def verify_vector_metadata(file_md5: str):
    """验证向量元数据"""
    return await knowledge_management_service.verify_vector_metadata_api(file_md5, vector_store_manager)

# ========== 目标检测接口 ==========
@app.post("/api/detect/objects")
async def object_detect(file: UploadFile = File(...)):
    """图片上传目标检测接口"""
    if detect_service is None:
        return ResponseWrapper.error(msg="目标检测服务初始化失败")
    try:
        content = await file.read()
        result = detect_service.detect(content)
        return ResponseWrapper.success(data=result)
    except Exception as e:
        app_logger.error(f"目标检测失败: {str(e)}")
        return ResponseWrapper.error(msg=f"检测失败: {str(e)}")

@app.get("/api/detect/objects-url")
async def object_detect_url(url: str):
    """URL图片目标检测接口"""
    if detect_service is None:
        return ResponseWrapper.error(msg="目标检测服务初始化失败")
    try:
        result = detect_service.detect_from_url(url)
        return ResponseWrapper.success(data=result)
    except Exception as e:
        app_logger.error(f"URL目标检测失败: {str(e)}")
        return ResponseWrapper.error(msg=f"检测失败: {str(e)}")

# ========== 图像处理接口 ==========
@app.post("/api/ai/remove_bg")
async def remove_background(file: UploadFile = File(...)):
    """智能去除图片背景接口"""
    if image_service is None:
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content=ResponseWrapper.error(code=503, msg="图像处理服务未启动").model_dump()
        )
    try:
        app_logger.info(f"开始去除图片背景: {file.filename}")
        input_image = await file.read()
        output_image = image_service.remove_background(input_image)
        app_logger.info(f"背景去除完成: {file.filename}")
        return Response(content=output_image, media_type="image/png")
    except Exception as e:
        error_msg = f"去除背景失败: {str(e)}"
        app_logger.error(error_msg)
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content=ResponseWrapper.error(msg=error_msg).model_dump()
        )

@app.post("/api/ai/face_blur")
async def face_blur(file: UploadFile = File(...)):
    """人脸打马赛克接口"""
    if image_service is None:
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content=ResponseWrapper.error(code=503, msg="图像处理服务未启动").model_dump()
        )
    try:
        app_logger.info(f"开始进行人脸打码: {file.filename}")
        input_image = await file.read()
        blurred_image = image_service.blur_faces(input_image)
        app_logger.info(f"人脸打码完成: {file.filename}")
        return Response(content=blurred_image, media_type="image/png")
    except Exception as e:
        error_msg = f"人脸打码失败: {str(e)}"
        app_logger.error(error_msg)
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content=ResponseWrapper.error(msg=error_msg).model_dump()
        )

@app.post("/api/ai/enhance_image")
async def enhance_image(file: UploadFile = File(...)):
    """增强图片清晰度接口"""
    if image_service is None:
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content=ResponseWrapper.error(code=503, msg="图像处理服务未启动").model_dump()
        )
    try:
        app_logger.info(f"开始增强图片清晰度: {file.filename}")
        input_image = await file.read()
        enhanced_image = image_service.enhance_image(input_image)
        app_logger.info(f"增强清晰度完成: {file.filename}")
        return Response(content=enhanced_image, media_type="image/jpeg")
    except Exception as e:
        error_msg = f"增强清晰度失败: {str(e)}"
        app_logger.error(error_msg)
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content=ResponseWrapper.error(msg=error_msg).model_dump()
        )

@app.post("/api/ai/change_background")
async def change_background(
    file: UploadFile = File(...), 
    background_image: Optional[UploadFile] = File(None),
    color: Optional[str] = None
):
    """智能更换图片背景接口"""
    if image_service is None:
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content=ResponseWrapper.error(code=503, msg="图像处理服务未启动").model_dump()
        )
    try:
        app_logger.info(f"开始更换图片背景: {file.filename}, 颜色: {color}, 背景图: {background_image.filename if background_image else 'None'}")
        input_image = await file.read()
        bg_image = None
        if background_image:
            bg_image = await background_image.read()
        output_image = image_service.change_background(
            input_image_bytes=input_image,
            background_color=color,
            background_image_bytes=bg_image
        )
        app_logger.info(f"背景更换完成: {file.filename}")
        return Response(content=output_image, media_type="image/png")
    except Exception as e:
        error_msg = f"更换背景失败: {str(e)}"
        app_logger.error(error_msg)
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content=ResponseWrapper.error(msg=error_msg).model_dump()
        )

# ========== 文本审核接口 ==========
@app.post("/api/comment/moderation")
async def comment_moderation(request: dict):
    """
    文本评论审核接口（纯 DFA 高性能匹配）
    请求体: { "comments": ["评论1", "评论2"], "mode": "fast|accurate|strict" }
    响应体: { "results": { "评论文本": {"label":0|1, "score":0.xx, "index":0} }, "costSeconds": 0.001 }
    """
    try:
        if not comment_moderation_service._ready:
            from starlette.responses import JSONResponse
            return JSONResponse(status_code=503, content={"error": "文本审核服务未就绪，请稍后重试"})

        comments = request.get('comments', [])
        mode = request.get('mode', 'accurate')

        if not comments or not isinstance(comments, list):
            from starlette.responses import JSONResponse
            return JSONResponse(status_code=400, content={"error": "comments 字段不能为空且必须是数组"})

        app_logger.info(f"文本审核请求 | mode={mode} | 评论数={len(comments)}")

        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            thread_pool,
            lambda: comment_moderation_service.moderate(comments, mode)
        )

        app_logger.info(f"文本审核完成 | costSeconds={res.get('costSeconds')}")
        return res
    except Exception as e:
        error_msg = f"文本审核失败: {str(e)}"
        app_logger.error(error_msg)
        return ResponseWrapper.error(msg=error_msg)


@app.post("/api/post/moderation")
async def post_moderation(request: dict):
    """
    AI 帖子内容审核接口
    请求体: { "title": "帖子标题", "content": "帖子内容" }
    响应体: { "safe": true/false, "reason": "..." }
    - safe=true: 内容安全，直接通过
    - safe=false: 确定违规，需要人工审核
    """
    try:
        title = request.get("title", "")
        content = request.get("content", "")
        if not title and not content:
            return ResponseWrapper.success(data={"safe": True, "reason": "内容为空，默认通过"})

        # 去除 HTML 标签，只保留纯文字
        import re
        clean_content = re.sub(r'<[^>]+>', '', content).strip()
        full_text = f"标题：{title}\n正文：{clean_content}"
        # 截断，防止过长
        full_text = full_text[:2000]

        from model.factory import create_chat_model
        model = create_chat_model()

        system_prompt = (
            "你是一个社区内容安全审核员，负责审核摄影/图库社区帖子。\n"
            "你的任务是判断帖子内容是否违反社区规定。\n"
            "以下类型的内容属于**确定违规**，必须拦截：\n"
            "  1. 色情、淫秽内容\n"
            "  2. 政治敏感、违法内容\n"
            "  3. 恶意攻击、辱骂他人\n"
            "  4. 诈骗、赌博、毒品相关\n"
            "  5. 大量垃圾广告信息\n"
            "普通的摄影技巧、生活感悟、情绪表达等均属于正常内容，应通过。\n"
            "模糊、边缘、不确定的内容，请放行（返回 safe=true）。\n"
            "只有你有99%的把握确定违规时，才返回 safe=false。\n\n"
            "你必须严格按照以下 JSON 格式返回，不要有任何多余文字：\n"
            '{"safe": true, "reason": "通过原因"}\n'
            '或\n'
            '{"safe": false, "reason": "违规原因"}'
        )

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请审核以下帖子内容：\n\n{full_text}")
        ]
        response = model.invoke(messages)
        raw = response.content.strip()
        app_logger.info(f"[帖子审核] LLM原始返回: {raw}")

        # 解析 JSON
        import json
        # 提取 JSON 块（防止模型在 JSON 前后加额外文字）
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            safe = bool(result.get("safe", True))
            reason = result.get("reason", "")
        else:
            # 解析失败，默认安全（宽松降级）
            safe = True
            reason = "解析模型返回失败，默认通过"
            app_logger.warning(f"[帖子审核] 无法解析LLM返回，默认放行: {raw}")

        app_logger.info(f"[帖子审核] 标题='{title[:30]}' safe={safe} reason={reason}")
        return ResponseWrapper.success(data={"safe": safe, "reason": reason})

    except Exception as e:
        error_msg = f"帖子审核失败: {str(e)}"
        app_logger.error(error_msg)
        # 异常时默认放行，不影响用户发帖
        return ResponseWrapper.success(data={"safe": True, "reason": f"审核服务异常，默认通过: {str(e)}"})


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Python RAG Service is running!",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "rag": "available" if rag_summarizer else "unavailable",
            "agent": "available" if agent else "unavailable",
            "vector_store": "available" if vector_store_manager else "unavailable",
            "object_detection": "available" if detect_service else "unavailable",
            "image_processing": "available" if image_service else "unavailable",
            "comment_moderation": "available" if comment_moderation_service._ready else "unavailable"
        },
        "memory_optimization": "disabled"
    }

# 主函数
if __name__ == "__main__":
    import uvicorn

    try:
        config = load_config()
        host = config.get('fastapi_host', '127.0.0.1')
        port = config.get('fastapi_port', 8001)

        app_logger.info(f"\n启动Python RAG Service")
        app_logger.info(f"地址: http://{host}:{port}")
        app_logger.info(f"Python版本: {sys.version[:5]}")
        app_logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        app_logger.info("=" * 80)

        profile = os.getenv('SPRING_PROFILES_ACTIVE', os.getenv('APP_PROFILES_ACTIVE', 'dev'))
        is_prod = (profile == 'prod')

        # 启动服务（生产环境下控制台静默，日志集中写入本地日志文件；开发环境下支持保存后自动热重启应用）
        if not is_prod:
            # 开启 uvicorn reload 模式，以便于在任意 py 工具脚本修改保存时，开发环境能够瞬间自动重载重启服务
            # 必须使用字符串路径 "main:app"，并动态加入 reload_dirs 的监测
            app_logger.info("[开发环境启动] 已经成功启用开发环境代码热更监控器(Reload-Monitor)！代码保存后将自动重起！")
            uvicorn.run(
                "main:app",
                host=host,
                port=port,
                log_level="debug",
                access_log=True,
                reload=True,
                reload_dirs=[os.path.dirname(__file__)]
                if os.path.exists(os.path.dirname(__file__))
                else [os.getcwd()]
            )
        else:
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="warning",  # 生产环境设置为 warning
                access_log=False  # 生产环境下禁用控制台 access_log 访问日志
            )
    except Exception as e:
        error_msg = f"启动服务失败: {str(e)}"
        app_logger.error(f"\n{error_msg}")
        app_logger.error(f"[SERVICE_START_FAILED] {error_msg}")
        sys.exit(1)