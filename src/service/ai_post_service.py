import json
import asyncio
from typing import AsyncGenerator
from utils.log_utils import app_logger
from model.factory import create_chat_model
from agent.ai_cover_generator_tool import generate_ai_cover
from agent.context import set_sa_token, get_sa_token
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class AIPostService:
    def __init__(self):
        pass

    async def process_ai_post_stream(self, request) -> AsyncGenerator[str, None]:
        """流式生成AI帖子：先流式生成正文，提取到标题后异步生成封面"""
        
        # 设置上下文 token，供后续的图片生成和上传使用
        if hasattr(request, 'sa_token') and request.sa_token:
            set_sa_token(request.sa_token)
            
        yield f"event: status\ndata: {json.dumps({'status': '正在撰写高质量正文...'})}\n\n"
        await asyncio.sleep(0.1)  # 确保状态发送
        
        system_prompt = (
            "你是一个专业的自媒体内容创作者，精通各平台（如小红书、微信公众号、社区论坛）的高爆款文案风格。\n"
            "请根据用户的提示词和分类，撰写一篇帖子。\n"
            "要求：\n"
            "1. 第一行必须是以 #TITLE# 开头的标题，例如：#TITLE# 标题内容（注意：标题内严禁包含任何 emoji 或特殊符号，必须是纯文字，以免封面渲染失败）。\n"
            "2. 第二行必须是以 #TAGS# 开头的话题标签，用逗号分隔，最多生成3个紧扣主题的话题，例如：#TAGS# 互联网,创业,职场\n"
            "3. 正文排版要求：必须是一篇结构完整、逻辑清晰、内容充实的文章！绝不能像写诗一样每说一句话就换行。每个段落必须有实质性的展开内容（建议每个段落不少于80字），段落与段落之间请使用连续的换行符（Enter）进行分隔。严禁在文本中使用任何 HTML 排版标签（如 <p>、<br>）或 Markdown 格式代码，保持纯净文本！\n"
            "4. 严禁在正文中生成或编造任何图片（切勿使用 <img> 标签或 markdown 图片语法），仅使用纯文本+emoji即可。\n"
            "5. 严禁在文章末尾生成任何类似 #标签名 的话题标签（hashtag），必须保持结尾干净。\n"
        )
        
        human_prompt_template = "请以用户的需求为绝对核心，写一篇关于“{prompt}”的帖子。（注：发布在“{category}”频道，仅供写作基调参考，切勿在正文中生硬提及该分类名称）"
        
        chat_model = create_chat_model()
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt_template)
        ])
        
        chain = prompt | chat_model | StrOutputParser()

        queue = asyncio.Queue()
        expected_tasks = 1
        tasks_completed = 0

        async def generate_cover_task(title: str):
            try:
                await queue.put(f"event: status\ndata: {json.dumps({'status': '正在绘制专属封面...'})}\n\n")
                
                token = get_sa_token()
                
                def run_cover_gen():
                    if token:
                        set_sa_token(token)
                    style_id = getattr(request, 'style_id', None)
                    return generate_ai_cover.invoke(
                        {"title": title[:20], "sub_title": "", "style_id": style_id}
                    )
                    
                cover_res_str = await asyncio.to_thread(run_cover_gen)
                cover_res = json.loads(cover_res_str)
                if "url" in cover_res:
                    await queue.put(f"event: cover\ndata: {json.dumps({'url': cover_res['url']})}\n\n")
            except Exception as e:
                app_logger.error(f"封面生成异常: {str(e)}")
            finally:
                await queue.put(("task_done",))

        async def generate_text_task():
            nonlocal expected_tasks
            try:
                title_buffer = ""
                title_extracted = False
                app_logger.info(f"========== [DEBUG] 开始流式生成帖子 ==========")
                app_logger.info(f"[DEBUG] 输入 prompt: {request.prompt}, category: {request.category}")
                
                chunk_count = 0
                async for chunk in chain.astream({
                    "prompt": request.prompt,
                    "category": request.category
                }):
                    chunk_count += 1
                    if chunk_count <= 3:
                        app_logger.info(f"[DEBUG] 收到第 {chunk_count} 个 chunk: {repr(chunk)}")
                    
                    if chunk:
                        await queue.put(f"event: content_chunk\ndata: {json.dumps({'text': chunk})}\n\n")
                        
                        if not title_extracted:
                            title_buffer += chunk
                            if "\n" in title_buffer:
                                first_line = title_buffer.split("\n")[0]
                                if "#TITLE#" in first_line:
                                    extracted_title = first_line.split("#TITLE#")[-1].strip()
                                    if extracted_title:
                                        expected_tasks += 1
                                        asyncio.create_task(generate_cover_task(extracted_title))
                                        title_extracted = True
                
                app_logger.info(f"========== [DEBUG] 流式生成完毕，总共收到 {chunk_count} 个 chunk ==========")
                                        
            except Exception as e:
                import traceback
                app_logger.error(f"正文生成异常: {str(e)}\n{traceback.format_exc()}")
                await queue.put(f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n")
            finally:
                app_logger.info(f"[DEBUG] 准备发送 task_done 信号")
                await queue.put(("task_done",))

        # 启动正文生成任务
        asyncio.create_task(generate_text_task())

        while True:
            item = await queue.get()
            if isinstance(item, tuple) and item[0] == "task_done":
                tasks_completed += 1
                if tasks_completed >= expected_tasks:
                    break
            else:
                yield item

        yield f"event: done\ndata: {json.dumps({'status': '生成完毕'})}\n\n"

ai_post_service = AIPostService()
