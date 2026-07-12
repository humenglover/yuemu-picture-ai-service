from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import time
import sys
import os
import re
import json

# 添加src目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from model.dto.rag_request_dto import RAGRequestDTO, StreamRAGRequestDTO
from model.vo.rag_vo import RAGResultVO, StreamChunkVO, StreamEndVO
from model.common.response_wrapper import ResponseWrapper
from model.common.constants import RAGConstants
from utils.log_utils import app_logger, knowledge_logger
from utils.file_utils import log_event
from rag.rag_summarize import RAGSummarizer
from agent.react_agent import ReactAgent
from agent.context import set_picture_metadata, clear_picture_metadata, set_session_id, set_sa_token

# ================= 预编译正则表达式 =================
CLEAN_JSON_RE = re.compile(r'\{[^{}]*\}')
CLEAN_MARKER_RE = re.compile(r'【[^】]*】[:：]?')
CLEAN_TOOL_PREFIX_RE = re.compile(r'(工具结果|搜索结果|时间结果|知识库信息|Action|Observation)[:：]')
PICTURE_METADATA_RE = re.compile(r'\[图片元数据:\s*(\{.+?\})\s*\]', re.DOTALL)
SNOWFLAKE_ID_RE = re.compile(r'\d{15,}')

class RAGService:
    """
    RAG服务类 - 处理RAG与Agent的动态路由及业务逻辑 (最佳实践瘦身版)
    """
    
    def __init__(self, rag_summarizer: RAGSummarizer, agent: Optional[ReactAgent] = None):
        self.rag_summarizer = rag_summarizer
        self.agent = agent
    
    # ================= 私有辅助方法 (DRY 原则重构) =================

    def _extract_and_set_picture_metadata(self, question: str) -> None:
        """提取并设置图片元数据到上下文"""
        clear_picture_metadata()
        if not question:
            return
            
        match = PICTURE_METADATA_RE.search(question)
        if not match:
            knowledge_logger.info('[METADATA_EXTRACT] 消息中未找到图片元数据')
            return
            
        try:
            metadata = json.loads(match.group(1))
            set_picture_metadata(metadata)
            knowledge_logger.info(f'[METADATA_EXTRACT] 成功提取: {json.dumps(metadata, ensure_ascii=False)}')
        except Exception as e:
            knowledge_logger.error(f'[METADATA_EXTRACT_ERROR] 解析异常: {str(e)}')

    def _build_agent_history(self, history: Optional[List[Dict[str, str]]], long_term_memory: Optional[str], user_persona: Optional[str] = None) -> List[Dict[str, str]]:
        """构建注入了长序列记忆和用户画像的 Agent 上下文"""
        agent_history = history.copy() if history else []
        
        # 组装注入的上下文（长时记忆 + 用户画像）
        injected_contexts = []
        if user_persona:
            injected_contexts.append(f"【当前用户信息】\n{user_persona}\n请基于该用户信息提供个性化回答。")
            
        if long_term_memory:
            injected_contexts.append(f"【平台数据历史上下文】\n{long_term_memory}\n请结合该超长上下文辅助回答。")
            
        if injected_contexts:
            combined_context = "\n\n".join(injected_contexts)
            knowledge_logger.info(f'[AGENT_CONTEXT_INJECT] 注入额外的上下文信息 | 长度: {len(combined_context)}')
            context_message = {
                "role": "user",
                "content": combined_context
            }
            # 插入到历史对话的最前面
            agent_history.insert(0, context_message)
        else:
            knowledge_logger.info('[AGENT_CONTEXT_INJECT] 无额外上下文信息注入')
            
        return agent_history

    def _format_stream_event(self, event_name: str, msg: str, data: dict, code: int = 200) -> str:
        """统一封装 SSE 流式返回格式"""
        return f"data: {json.dumps({'event': event_name, 'data': {'code': code, 'msg': msg, 'data': data}}, ensure_ascii=False)}\n\n"

    def _append_tool_enforcement(self, question: str) -> str:
        """物理层级的提示词劫持：在用户的最后一条消息末尾强行注入规则，对抗大模型的遗忘机制"""
        enforcement_suffix = "\n\n【系统最高级底层指令】：你必须且只能通过调用工具（如果无需查数据，请必须调用 `think` 工具）来回应，绝对禁止直接回复文本内容！"
        return question.strip() + enforcement_suffix

    # ================= 核心路由与清洗逻辑 =================



    def clean_agent_response(self, raw_response: str, question: str) -> str:
        """高阶清理器：抹除大模型幻觉与 Agent 工具痕迹并强制限制标题级别最高为二级标题"""
        if not raw_response:
            return "抱歉，由于系统或网络波动，暂未获取到有效信息，请换个说法重试。"
            
        resp = raw_response.strip()
        # 彻底拦截 3 个及以上的连续 '#' 符号，全强制替换为 2 个 '#' 符号，防止三级及更深标题污染前端
        resp = re.sub(r'#{3,}', '##', resp)
        
        resp = CLEAN_JSON_RE.sub('', resp)
        resp = CLEAN_MARKER_RE.sub('', resp)
        resp = CLEAN_TOOL_PREFIX_RE.sub('', resp)
        
        q_clean = question.strip()
        if q_clean and q_clean in resp and resp != q_clean:
            resp = resp.replace(q_clean, "").strip()
            
        resp = resp.lstrip('：:！？。,， \n').strip()
        
        # 过滤日志行
        skip_words = ('type:', 'query:', 'result:', 'error:', 'observation:')
        lines = [line.strip() for line in resp.split('\n') if line.strip() and not any(sw in line.lower() for sw in skip_words)]
        resp = '\n'.join(lines).strip()
        
        if not resp or resp == "已为您分析相关信息，但暂未获取到有效结果":
            return "抱歉，暂未获取到有效信息，请换个说法重试。"

        # 去除末尾孤立的单个 * （Qwen3 no_think 模式偶发的尾部冗余标记，非 ** 粗体）
        if resp.endswith('*') and not resp.endswith('**'):
            resp = resp[:-1].rstrip()

        return resp

    # ================= 业务处理接口 =================

    def process_rag_sync(self, request: RAGRequestDTO) -> ResponseWrapper:
        """同步 RAG 请求处理"""
        session_id = request.session_id or f"session_{int(time.time())}"
        safe_q = (request.question[:50] + "...") if request.question else ""
        
        if not request.question or not request.question.strip():
            return ResponseWrapper.bad_request("问题不能为空", {"answer": "", "session_id": session_id})

        try:
            app_logger.info(f"开始同步RAG | session: {session_id} | q: {safe_q}")
            set_session_id(session_id)
            set_sa_token(request.sa_token)
            self._extract_and_set_picture_metadata(request.question)

            # 全量请求直接交由 Agent 处理
            total_tokens = 0
            if self.agent:
                app_logger.info(f"Agent接管请求 | session: {session_id}")
                agent_history = self._build_agent_history(request.history, request.long_term_memory, request.user_persona)
                app_logger.info("[AGENT] 知识库检索完成，开始通过 Agent 处理同步请求")
                enforced_question = self._append_tool_enforcement(request.question)
                agent_res = self.agent.execute(
                    enforced_question, 
                    agent_history, 
                    model=request.model,
                    sa_token=request.sa_token,
                    session_id=session_id
                )
                if isinstance(agent_res, dict):
                    raw_ans = agent_res.get("answer", "")
                    total_tokens = agent_res.get("total_tokens", 0)
                else:
                    raw_ans = agent_res
                
                final_answer = self.clean_agent_response(raw_ans, request.question)
                
                if not final_answer or final_answer == request.question.strip():
                    final_answer = "抱歉，处理该请求时遇到了点小问题，请稍后再试。"
            else:
                # 兜底：如果没配置 Agent，回退到原始 RAG
                final_answer = self.rag_summarizer.summarize_with_rag(
                    question=request.question,
                    history=request.history,
                    long_term_memory=request.long_term_memory
                )

            result_vo = RAGResultVO(
                answer=final_answer, session_id=session_id, 
                top_k=request.top_k, temperature=request.temperature,
                total_tokens=total_tokens
            )
            return ResponseWrapper.success(data=result_vo.model_dump())

        except Exception as e:
            app_logger.error(f"同步RAG失败 | session: {session_id} | err: {str(e)}")
            return ResponseWrapper.bad_request("服务异常，请稍后再试", {"answer": "", "session_id": session_id})

    def process_summarize(self, request: RAGRequestDTO) -> ResponseWrapper:
        session_id = request.session_id or f"sum_{int(time.time())}"
        try:
            return ResponseWrapper.success(data=RAGResultVO(
                answer=self.rag_summarizer.direct_summarize(request.question), 
                session_id=session_id
            ).model_dump())
        except Exception as e:
            app_logger.error(f"摘要处理失败: {str(e)}")
            return ResponseWrapper.bad_request("摘要处理异常")

    async def process_summarize_async(self, request: RAGRequestDTO) -> ResponseWrapper:
        session_id = request.session_id or f"sum_{int(time.time())}"
        try:
            summary = await self.rag_summarizer.direct_summarize_async(request.question)
            return ResponseWrapper.success(data=RAGResultVO(answer=summary, session_id=session_id).model_dump())
        except Exception as e:
            app_logger.error(f"异步摘要处理失败: {str(e)}")
            return ResponseWrapper.bad_request("摘要处理异常")

    def process_rag_stream(self, request: StreamRAGRequestDTO) -> Generator[str, None, None]:
        """流式 RAG 请求处理"""
        session_id = request.session_id or f"session_{int(time.time())}"
        
        if not request.question or not request.question.strip():
            log_event('STREAM_QUESTION_INVALID', '无效流式请求', {'session_id': session_id})
            yield self._format_stream_event("error", "问题不能为空", {"answer": "", "session_id": session_id}, 400)
            return

        log_event('STREAM_START', '开始处理流式RAG', {'session_id': session_id})
        set_session_id(session_id)
        set_sa_token(request.sa_token)
        self._extract_and_set_picture_metadata(request.question)
        
        full_answer = ""
        total_tokens = 0
        try:
            yield self._format_stream_event("start", "开始流式传输", {
                "session_id": session_id, "top_k": request.top_k, "temperature": request.temperature
            })

            # 流式完全交给 Agent
            if self.agent:
                agent_history = self._build_agent_history(request.history, request.long_term_memory, request.user_persona)
                app_logger.info("[AGENT] 知识库检索完成，开始通过 Agent 处理流式请求")
                enforced_question = self._append_tool_enforcement(request.question)
                stream_generator = self.agent.execute_stream(
                    enforced_question, 
                    agent_history, 
                    model=request.model,
                    sa_token=request.sa_token,
                    session_id=session_id
                )
            else:
                stream_generator = self.rag_summarizer.stream_summarize_with_rag(
                    question=request.question, history=request.history, long_term_memory=request.long_term_memory
                )

            for chunk in stream_generator:
                if not chunk:
                    continue
                    
                if chunk.startswith("__STATUS__:"):
                    yield self._format_stream_event("status", "AI推导进度", {
                        "status": chunk.replace("__STATUS__:", "").strip(), "session_id": session_id
                    })
                    continue
                    
                if chunk.startswith("__TOKEN_USAGE__:"):
                    try:
                        total_tokens = int(chunk.replace("__TOKEN_USAGE__:", "").strip())
                    except ValueError:
                        pass
                    continue

                if chunk != request.question.strip():
                    full_answer += chunk
                    yield self._format_stream_event("chunk", "流式数据块", {
                        "chunk": chunk, "full_answer": full_answer, "session_id": session_id
                    })
            
            # 清洗最终完整答案（去除 Qwen3 no_think 模式偶发的尾部孤立 * 标记）
            if full_answer.endswith('*') and not full_answer.endswith('**'):
                full_answer = full_answer[:-1].rstrip()

            yield self._format_stream_event("end", "传输完成", {
                "answer": full_answer, "session_id": session_id,
                "answer_length": len(full_answer), "total_tokens": total_tokens
            })

        except Exception as e:
            app_logger.error(f"流式处理异常 | session: {session_id} | err: {str(e)}")
            yield self._format_stream_event("error", "服务异常", {
                "answer": "", "session_id": session_id, "error_detail": str(e)
            }, 500)