from typing import List, Dict, Any, Optional
import os
import re
from langchain.agents import create_agent
from model.factory import create_chat_model
from utils.log_utils import knowledge_logger
from .tools import available_tools

class ReactAgent:
    def __init__(self):
        self._agents = {}
        # 初始化默认Agent
        self.get_agent()
            
        knowledge_logger.info('[AGENT_INIT] ReactAgent 初始化完成（支持多模型动态切换）')

    def get_agent(self, model_name: Optional[str] = None):
        key = model_name or "default"
        if key not in self._agents:
            chat_model = create_chat_model(model_name)
            prompt_file_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'agent_system_prompt.txt')
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
                
            agent = create_agent(
                model=chat_model,
                tools=available_tools,
                system_prompt=system_prompt,
            )
            self._agents[key] = agent
        return self._agents[key]

    def _sanitize_text(self, text: str) -> str:
        if not text:
            return ""
        if '###' in text:
            text = text.replace('### ', '## ')
            text = text.replace('###', '##')
        if '## ' in text and not text.startswith('## '):
            text = text.replace('## ', '\n## ')
        return text

    def _extract_response_from_result(self, result) -> str:
        """从 LangGraph invoke 结果中提取文本（兼容新旧格式）"""
        # LangGraph 格式: {"messages": [HumanMessage, ..., AIMessage]}
        if "messages" in result:
            msgs = result["messages"]
            if msgs:
                last = msgs[-1]
                content = last.content if hasattr(last, 'content') else str(last)
                if isinstance(content, list):  # 多模态内容块
                    texts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
                    return " ".join(t for t in texts if t)
                return str(content)
        # 旧版 AgentExecutor 格式: {"output": "..."}
        if "output" in result:
            output = result["output"]
            if isinstance(output, list) and output:
                return output[-1].content
            return str(output)
        return ""

    def execute(self, query: str, history: Optional[List[Dict[str, str]]] = None, model: Optional[str] = None, sa_token: Optional[str] = None, session_id: Optional[str] = None):
        try:
            knowledge_logger.info(f'[AGENT_EXECUTE] 开始执行查询 | query: {query} | model: {model}')
            messages = []
            if history:
                for msg in history:
                    role = msg.get("role", "user")
                    messages.append({"role": role, "content": msg.get("content", "")})
            messages.append({"role": "user", "content": query})
            input_dict = {"messages": messages}
            
            agent = self.get_agent(model)
            config = {
                "configurable": {
                    "sa_token": sa_token,
                    "session_id": session_id
                }
            }
            result = agent.invoke(input_dict, config=config)
            
            response = self._extract_response_from_result(result)
            return self._sanitize_text(response)
        except Exception as e:
            error_msg = str(e)
            knowledge_logger.error(f'[AGENT_EXECUTE_ERROR] {error_msg}')
            return f"抱歉，系统暂时开小差了，请稍后再试。"

    def execute_stream(self, query: str, history: Optional[List[Dict[str, str]]] = None, model: Optional[str] = None, sa_token: Optional[str] = None, session_id: Optional[str] = None):
        try:
            knowledge_logger.info(f'[AGENT_EXECUTE_STREAM] 开始流式执行查询 | query: {query} | model: {model}')
            messages = []
            if history:
                for msg in history:
                    messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            messages.append({"role": "user", "content": query})
            input_dict = {"messages": messages}
            
            import json
            knowledge_logger.info(f'[AGENT_REQUEST_TO_AI] ========== 开始打印完整请求 ==========')
            knowledge_logger.info(f'[AGENT_REQUEST_TO_AI] 消息总数: {len(messages)}')
            for idx, msg in enumerate(messages):
                knowledge_logger.info(f'[AGENT_REQUEST_TO_AI] ========== 消息 [{idx}] ==========')
                knowledge_logger.info(f'[AGENT_REQUEST_TO_AI] Role: {msg.get("role")}')
                knowledge_logger.info(f'[AGENT_REQUEST_TO_AI] Content (完整): {msg.get("content")}')
            knowledge_logger.info(f'[AGENT_REQUEST_TO_AI] ========== 完整请求JSON ==========')
            knowledge_logger.info(f'[AGENT_REQUEST_TO_AI] {json.dumps(input_dict, ensure_ascii=False, indent=2)}')
            knowledge_logger.info(f'[AGENT_REQUEST_TO_AI] ========== 请求打印结束 ==========')
            
            TOOL_DISPLAY_NAMES = {
                "tavily_search_tool": "全网高速检索 (Tavily)",
                "tavily_search": "全网检索",
                "weather_query_tool": "实时天气查询",
                "search_pictures": "站内图库智能检索",
                "search_pictures_by_image": "以图搜图智能检索",
                "search_pexels_images": "商用高清图库检索",
                "get_pexels_curated": "高清精选图集获取",
                "get_current_time": "时间校准",
                "datetime_tool": "时间校准",
                "search_site": "全站数据综合检索",
                "nanodet_object_detection": "视觉目标检测（NanoDet）",
                "nanodet_detect": "视觉目标检测（NanoDet）",
                "extract_image_color_palette": "色板分析引擎",
                "extract_image_exif": "EXIF 参数解析",
                "classify_image_tool": "图像深度特征分析",
                "rag_summarize": "平台知识库检索",
                "tts_reply": "语音合成系统",
                "list_available_spaces": "空间资源查询",
                "upload_picture_to_space": "网络图片上传",
                "upload_local_image": "本地图片上传",
                "smart_crop_and_upload": "智能裁剪引擎",
                "enhance_and_upload": "画质增强引擎",
                "add_watermark_and_upload": "版权水印系统",
                "apply_filter_and_upload": "氛围感滤镜大师",
                "apply_frame_and_upload": "艺术装裱大师",
                "generate_art_card_and_upload": "社交卡片智造大师",
                "generate_image": "AI 画图引擎",
                "generate_ai_cover": "AI 智能封面生成器",
                "delete_picture": "图片删除服务",
                "check_system_status": "系统状态监控",
                "get_top_processes": "进程资源分析",
                "get_my_personal_data": "个人中心数据检索",
                "get_picture_detail_data": "图片详情档案",
                "get_post_detail_data": "帖子详情档案",
                "get_user_detail_data": "用户公开档案",
                "get_follow_or_fan_list_data": "社交关系网络检索",
                "think": "深度意图推理",
                "noop": "对话意图重定向",
                "get_session_history": "会话历史记忆检索",
                "search_long_term_memory": "超长历史记忆搜索"
            }

            emitted_tool_ids = set()
            tool_result_emitted = False
            agent = self.get_agent(model)
            # 获取 agent 中实际绑定的模型名
            try:
                actual_model = agent.nodes["model"].bound.model_name
            except Exception:
                actual_model = model or "default(config)"
            knowledge_logger.info(f'[AGENT_MODEL_USED] 本次请求使用模型: {actual_model} | 请求模型参数: {model}')
            event_count = 0
            
            config = {
                "configurable": {
                    "sa_token": sa_token,
                    "session_id": session_id
                }
            }

            # 使用 stream_mode="messages" 获取 token 级流式输出
            # 返回 (message_chunk, metadata) 元组
            for chunk, metadata in agent.stream(input_dict, stream_mode="messages", config=config):
                event_count += 1
                chunk_type = type(chunk).__name__

                # 跳过人类消息
                if 'HumanMessage' in chunk_type:
                    continue

                # 提取工具调用信息
                tcalls = []
                try:
                    if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                        tcalls = chunk.tool_call_chunks
                    elif hasattr(chunk, 'tool_calls') and getattr(chunk, 'tool_calls', None):
                        tcalls = chunk.tool_calls
                except Exception:
                    pass

                if tcalls:
                    for tc in tcalls:
                        if isinstance(tc, dict):
                            tid = tc.get('id') or ''
                            tname = tc.get('name') or ''
                        else:
                            tid = getattr(tc, 'id', '') or ''
                            tname = getattr(tc, 'name', '') or ''

                        if not tname or not tid or tid in emitted_tool_ids:
                            continue
                        emitted_tool_ids.add(tid)
                        display_name = TOOL_DISPLAY_NAMES.get(tname, tname)
                        yield f"__STATUS__:正在调动 [{display_name}] 引擎...\n"
                    continue

                # 工具结果消息：发出"分析完成"状态
                if 'ToolMessage' in chunk_type:
                    if not tool_result_emitted:
                        tool_result_emitted = True
                        yield "__STATUS__:数据抓取完毕，正在汇总核心信息...\n"
                    continue

                # AIMessageChunk 正文 token
                content = getattr(chunk, 'content', None)
                if content and isinstance(content, str):
                    if not content.startswith(('{', 'Action:', 'Observation:', 'Thought:')):
                        clean_content = self._sanitize_text(content)
                        if clean_content:
                            yield clean_content

            knowledge_logger.info(f'[AGENT_STREAM_DONE] 流式处理完成 | chunk总数: {event_count}')

        except Exception as e:
            error_str = str(e)
            if "AuthenticationError" in error_str:
                error_msg = f"API密钥无效或已过期，请检查配置。"
            elif "ConnectionError" in error_str or "Timeout" in error_str:
                error_msg = f"连接大模型服务超时，请检查网络。"
            elif "InvalidParameter" in error_str or "JSON format" in error_str:
                error_msg = "智能体底层逻辑构建异常，请换个说法重试。"
            else:
                error_msg = f"流式调用中断，请稍后重试。"

            knowledge_logger.error(f'[AGENT_EXECUTE_STREAM_ERROR] 流式处理失败: {error_str}')
            yield f"抱歉，系统暂时遇到小问题，请稍后重试。"