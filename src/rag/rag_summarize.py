import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.factory import create_chat_model
from utils.log_utils import rag_logger
from utils.file_utils import log_event
from rag.bm25_retriever import BM25Retriever
from rag.rrf_fusion import RRFFusion

class RAGSummarizer:
    def __init__(self, vector_store, enable_hybrid_search: bool = True):
        """
        初始化 RAG 摘要器（支持 Hybrid Search：向量 + BM25 + RRF 融合）
        
        Args:
            vector_store: 向量存储
            enable_hybrid_search: 是否启用混合检索（默认 True）
        """
        self.chat_model = create_chat_model()
        self.vector_store = vector_store
        self.retriever = vector_store.get_retriever()
        self.prompt_template = self.load_prompt_template()
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        self.chain = self._init_chain()
        # 预建摘要 PromptTemplate（避免每次调用重复创建）
        self._summary_prompt = PromptTemplate.from_template(
            "请简要总结以下对话内容（100字以内）：\n\n{text}\n\n回答："
        )
        
        # ========== Hybrid Search 核心模块（纯 BM25 + 向量融合）==========
        self.enable_hybrid_search = enable_hybrid_search
        self.bm25_retriever = None
        self.rrf_fusion = None
        
        if self.enable_hybrid_search:
            try:
                # 初始化 RRF 融合器
                self.rrf_fusion = RRFFusion(k=60)
                # BM25 使用懒加载：空索引，第一次检索时按需填充，避免全量文档常驻内存
                self.bm25_retriever = BM25Retriever(documents=[])
                rag_logger.info("[RAG_INIT] Hybrid Search 架构初始化完成（向量 + BM25 懒加载 + RRF 融合）")
            except Exception as e:
                rag_logger.error(f"[RAG_INIT_ERROR] Hybrid Search 初始化失败，降级为纯向量检索: {str(e)}")
                self.enable_hybrid_search = False
    
    def _init_chain(self):
        chain = self.prompt | self.chat_model | StrOutputParser()
        return chain
    
    def _init_bm25_retriever(self):
        """已废弃：BM25 改为懒加载，此方法保留仅供外部兼容调用，实际不再执行全量加载"""
        rag_logger.info("[BM25_INIT] BM25 使用懒加载模式，跳过全量文档加载，首次检索时自动填充")
        self.bm25_retriever = BM25Retriever(documents=[])
    
    def load_prompt_template(self):
        """加载提示词模板"""
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'rag_summarize_prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def retrieve_docs(self, query: str, top_k: int = 5) -> List[Document]:
        """
        检索文档（支持 Hybrid Search：向量 + BM25 + RRF 融合）
        
        流程：
        1. 向量检索 (语义) → top 50
        2. BM25 检索 (关键词) → top 50
        3. RRF 融合 → 去重排序 → top_k
        
        Args:
            query: 查询文本
            top_k: 最终返回的文档数量
            
        Returns:
            检索到的文档列表
        """
        rag_logger.info(f"[RETRIEVE_DOCS] ========== 开始文档检索 ==========")
        rag_logger.info(f"[RETRIEVE_DOCS] 用户查询: {query[:100]}...")
        rag_logger.info(f"[RETRIEVE_DOCS] 目标返回数量: {top_k}")
        
        # 如果未启用混合检索，使用原始向量检索
        if not self.enable_hybrid_search or not self.bm25_retriever or not self.rrf_fusion:
            rag_logger.info("[RETRIEVE_DOCS] 使用纯向量检索模式")
            docs = self.retriever.invoke(query)
            rag_logger.info(f"[RETRIEVE_DOCS] 向量检索完成 | 召回数量: {len(docs)}")
            return docs[:top_k]
        
        try:
            # ========== Hybrid Search 混合检索 ==========
            rag_logger.info("[RETRIEVE_DOCS] ========== Hybrid Search 混合检索 ==========")
            
            # 1. 向量检索（召回 50 条）
            vector_top_k = 50
            rag_logger.info(f"[RETRIEVE_DOCS] [1/3] 开始向量检索 | 目标召回: {vector_top_k}")
            vector_docs = self.retriever.invoke(query)
            
            # 转换为 (doc, score) 格式
            vector_results = []
            for i, doc in enumerate(vector_docs[:vector_top_k]):
                # 使用倒数排名作为分数
                score = 1.0 / (i + 1)
                vector_results.append((doc, score))
            
            rag_logger.info(f"[RETRIEVE_DOCS] [1/3] 向量检索完成 | 实际召回: {len(vector_results)}")
            
            # 2. BM25 检索（懒加载：索引为空时用向量检索结果填充，避免全量加载）
            bm25_top_k = 50
            rag_logger.info(f"[RETRIEVE_DOCS] [2/3] 开始 BM25 关键词检索 | 目标召回: {bm25_top_k}")
            
            # 懒加载：仅在索引为空时用本次向量检索结果初始化，避免全量加载
            if self.bm25_retriever is not None and not self.bm25_retriever.documents:
                rag_logger.info("[RETRIEVE_DOCS] BM25 索引为空（懒加载），用本次向量结果填充")
                self.bm25_retriever.update_documents(vector_docs)
            
            bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_top_k) if self.bm25_retriever else []
            rag_logger.info(f"[RETRIEVE_DOCS] [2/3] BM25 检索完成 | 实际召回: {len(bm25_results)}")
            
            # 3. RRF 融合
            rag_logger.info(f"[RETRIEVE_DOCS] [3/3] 开始 RRF 融合")
            fused_results = self.rrf_fusion.fuse(
                vector_results=vector_results,
                bm25_results=bm25_results,
                vector_weight=0.5,
                bm25_weight=0.5
            )
            rag_logger.info(f"[RETRIEVE_DOCS] [3/3] RRF 融合完成 | 融合后数量: {len(fused_results)}")
            
            # 提取文档（去掉分数）并取 top_k
            final_docs = [doc for doc, score in fused_results[:top_k]]
            
            # ========== 检索流程总结 ==========
            rag_logger.info(f"[RETRIEVE_DOCS] ========== 检索流程总结 ==========")
            rag_logger.info(f"[RETRIEVE_DOCS] 向量检索: {len(vector_results)} 条")
            rag_logger.info(f"[RETRIEVE_DOCS] BM25 检索: {len(bm25_results)} 条")
            rag_logger.info(f"[RETRIEVE_DOCS] RRF 融合: {len(fused_results)} 条")
            rag_logger.info(f"[RETRIEVE_DOCS] 最终送入 LLM: {len(final_docs)} 条")
            rag_logger.info(f"[RETRIEVE_DOCS] ========== 检索完成 ==========")
            
            return final_docs
            
        except Exception as e:
            rag_logger.error(f"[RETRIEVE_DOCS_ERROR] Hybrid Search 失败，降级为纯向量检索: {str(e)}")
            import traceback
            rag_logger.error(traceback.format_exc())
            
            # 降级：使用纯向量检索
            docs = self.retriever.invoke(query)
            rag_logger.info(f"[RETRIEVE_DOCS] 降级向量检索完成 | 召回数量: {len(docs)}")
            return docs[:top_k]
    
    def summarize_with_rag(self, 
                          question: str, 
                          history: List[Dict[str, str]] = None,
                          long_term_memory: str = "") -> str:
        """使用RAG进行摘要"""
        try:
            # 记录AI回答开始事件
            safe_question = question[:50]
            history_len = len(history or [])
            ltm_len = len(long_term_memory) if long_term_memory else 0
            rag_logger.info('[AI_ANSWER_START] 开始AI回答 | question: ' + safe_question + '... | history_length: ' + str(history_len) + ' | long_term_memory_length: ' + str(ltm_len))
            
            # 打印接收到的long_term_memory内容（包含ES搜索结果）
            if long_term_memory:
                rag_logger.info('[LONG_TERM_MEMORY] 接收到上下文信息:\n' + long_term_memory)
            else:
                rag_logger.info('[LONG_TERM_MEMORY] 未接收到上下文信息')
            
            # 检索相关文档
            context_docs = self.retrieve_docs(question)
            
            # 构建上下文字符串
            context = self.build_context_str(context_docs, long_term_memory)
            
            # 构建历史对话字符串
            history_str = self.build_history_str(history or [])
            
            # 调用链
            result = self.chain.invoke({
                "input": question,
                "context": context,
                "history": history_str
            })
            
            # 记录AI回答完成事件
            rag_logger.info('[AI_ANSWER_COMPLETE] AI回答完成 | question: ' + safe_question + '... | answer_length: ' + str(len(result)))
            
            return result
                
        except Exception as e:
            err_info = str(e)
            print("RAG摘要过程中出现错误: " + err_info)
            # 记录AI回答错误事件
            safe_question = question[:50]
            rag_logger.error('[AI_ANSWER_ERROR] AI回答过程中出现错误 | question: ' + safe_question + '... | error: ' + err_info)
            return "抱歉，AI服务暂时不可用，请稍后再试。错误详情: " + err_info
    
    def direct_summarize(self, text: str, prompt_template: str = None) -> str:
        """直接对文本进行摘要，不使用 RAG 检索"""
        try:
            if not text:
                rag_logger.warning('[DIRECT_SUMMARIZE] 输入文本为空')
                return ""
            
            # 优先使用预构建的摘要 PromptTemplate，避免每次调用重复创建
            if prompt_template:
                summary_prompt = PromptTemplate.from_template(prompt_template)
            else:
                summary_prompt = self._summary_prompt
            
            # 使用较调皮的日志记录输入前 50 个字符
            rag_logger.info('[DIRECT_SUMMARIZE_START] 开始摘要 | 文本预览: ' + text[:50].replace('\n', ' ') + '...')
            
            # 尝试直接调用模型获取原始输出
            raw_response = self.chat_model.invoke(summary_prompt.format(text=text))
            
            # 获取解析后的文本
            result = ""
            if hasattr(raw_response, 'content'):
                result = raw_response.content
            else:
                result = str(raw_response)
            
            result = result.strip()
            
            if not result:
                rag_logger.warning('[DIRECT_SUMMARIZE_EMPTY] 模型返回了空字符串 | 原始响应: ' + str(raw_response))
            else:
                rag_logger.info('[DIRECT_SUMMARIZE_COMPLETE] 摘要完成 | 摘要长度: ' + str(len(result)) + ' | 摘要预览: ' + result[:30] + '...')
            
            return result
        except Exception as e:
            rag_logger.error('[DIRECT_SUMMARIZE_ERROR] 摘要失败: ' + str(e))
            import traceback
            rag_logger.error(traceback.format_exc())
            return ""

    async def direct_summarize_async(self, text: str, prompt_template: str = None) -> str:
        """异步直接对文本进行摘要，不使用 RAG 检索"""
        try:
            if not text:
                rag_logger.warning('[DIRECT_SUMMARIZE_ASYNC] 输入文本为空')
                return ""
            
            # 使用简单的摘要模板，避免知识库干扰
            if not prompt_template:
                prompt_template = "请简要总结以下对话内容（100字以内）：\n\n{text}\n\n回答："
            
            summary_prompt = PromptTemplate.from_template(prompt_template)
            
            rag_logger.info('[DIRECT_SUMMARIZE_ASYNC_START] 开始异步摘要 | 文本预览: ' + text[:50].replace('\n', ' ') + '...')
            
            # 使用 ainvoke 异步调用模型
            raw_response = await self.chat_model.ainvoke(summary_prompt.format(text=text))
            
            # 获取解析后的文本
            result = ""
            if hasattr(raw_response, 'content'):
                result = raw_response.content
            else:
                result = str(raw_response)
            
            result = result.strip()
            
            if not result:
                rag_logger.warning('[DIRECT_SUMMARIZE_ASYNC_EMPTY] 模型返回了空字符串 | 原始响应: ' + str(raw_response))
            else:
                rag_logger.info('[DIRECT_SUMMARIZE_ASYNC_COMPLETE] 异步摘要完成 | 摘要长度: ' + str(len(result)) + ' | 摘要预览: ' + result[:30] + '...')
            
            return result
        except Exception as e:
            rag_logger.error('[DIRECT_SUMMARIZE_ASYNC_ERROR] 异步摘要失败: ' + str(e))
            import traceback
            rag_logger.error(traceback.format_exc())
            return ""

    def build_context_str(self, context_docs: List[Document], long_term_memory: str = "") -> str:
        """构建上下文字符串"""
        context_parts = []
        if long_term_memory:
            context_parts.append("【历史相关记忆】\n" + long_term_memory)
        
        if context_docs:
            context_parts.append("【参考知识库】")
        for i, doc in enumerate(context_docs, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            source = doc.metadata.get('source', '未知来源') if hasattr(doc, 'metadata') else '未知来源'
            # 纯字符串拼接，删除f-string
            context_parts.append(str(i) + ". 来源: " + source + " | 内容: " + content)
        
        return "\n".join(context_parts)
    
    def build_history_str(self, history: List[Dict[str, str]]) -> str:
        """构建历史对话字符串"""
        if not history:
            return ""
        
        history_parts = ["【会话历史】"]
        for item in history:
            role = "用户" if item.get("role") == "user" else "客服"
            content = item.get("content", "")
            # 纯字符串拼接，删除f-string
            history_parts.append(role + "：" + content)
        
        return "\n".join(history_parts)
    
    def stream_summarize_with_rag(self, 
                                 question: str, 
                                 history: List[Dict[str, str]] = None,
                                 long_term_memory: str = ""):
        """流式使用RAG进行摘要"""
        try:
            # 记录流式AI回答开始事件
            safe_question = question[:50]
            history_len = len(history or [])
            ltm_len = len(long_term_memory) if long_term_memory else 0
            rag_logger.info('[STREAM_AI_ANSWER_START] 开始流式AI回答 | question: ' + safe_question + '... | history_length: ' + str(history_len) + ' | long_term_memory_length: ' + str(ltm_len))
            
            # 打印接收到的long_term_memory内容（包含ES搜索结果）
            if long_term_memory:
                rag_logger.info('[LONG_TERM_MEMORY] 接收到上下文信息:\n' + long_term_memory)
            else:
                rag_logger.info('[LONG_TERM_MEMORY] 未接收到上下文信息')
            
            # 检索相关文档
            context_docs = self.retrieve_docs(question)
            
            # 构建上下文字符串
            context = self.build_context_str(context_docs, long_term_memory)
            
            # 构建历史对话字符串
            history_str = self.build_history_str(history or [])
            
            # 流式调用链
            for chunk in self.chain.stream({
                "input": question,
                "context": context,
                "history": history_str
            }):
                yield chunk
                    
        except Exception as e:
            err_info = str(e)
            print("RAG流式摘要过程中出现错误: " + err_info)
            # 记录流式AI回答错误事件
            safe_question = question[:50]
            rag_logger.error('[STREAM_AI_ANSWER_ERROR] 流式AI回答过程中出现错误 | question: ' + safe_question + '... | error: ' + err_info)
            yield "抱歉，AI流式服务暂时不可用，请稍后再试。错误详情: " + err_info
    
    async def astream_summarize_with_rag(self, 
                                        question: str, 
                                        history: List[Dict[str, str]] = None):
        """异步流式使用RAG进行摘要"""
        try:
            # 检索相关文档
            context_docs = self.retrieve_docs(question)
            
            # 构建上下文字符串
            context = self.build_context_str(context_docs)
            
            # 构建历史对话字符串
            history_str = self.build_history_str(history or [])
            
            # 异步流式调用链
            async for chunk in self.chain.astream({
                "input": question,
                "context": context,
                "history": history_str
            }):
                yield chunk
                    
        except Exception as e:
            err_info = str(e)
            print("RAG异步流式摘要过程中出现错误: " + err_info)
            yield "抱歉，AI异步流式服务暂时不可用，请稍后再试。错误详情: " + err_info