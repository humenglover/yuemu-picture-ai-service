from langchain_core.tools import tool
from utils.log_utils import knowledge_logger

# 延迟导入辅助函数
def get_rag_service():
    try:
        import main
        if hasattr(main, 'rag_summarizer') and main.rag_summarizer is not None:
            return main.rag_summarizer
    except Exception:
        pass
    
    # 备用兜底机制：延迟构建初始化 RAGSummarizer 实例
    from rag.rag_summarize import RAGSummarizer
    from rag.vector_store import VectorStoreManager
    vector_store = VectorStoreManager()
    return RAGSummarizer(vector_store)

@tool(description="获取关于悦木图库平台（Yuemu）的官方网站、联系方式、平台规则、功能指南、会员机制、服务条款等站内专有知识。凡是用户询问本平台自身相关的信息，必须优先调用此工具进行检索，严禁编造（幻觉）。")
def rag_summarize(query: str) -> str:
    """使用本地RAG服务检索相关知识片段（仅检索，不生成，由 Agent 统一综合回答）"""
    try:
        rag_service = get_rag_service()
        # 只做向量检索，不调用 LLM 生成
        # 避免：RAGSummarizer 内部 LLM 生成一次 + Agent 最终 LLM 再生成一次 → 双重重复
        docs = rag_service.retrieve_docs(query, top_k=8)
        if not docs:
            knowledge_logger.info(f'[RAG_TOOL] 未检索到相关文档 | query: {query}')
            return "知识库中未找到相关信息。"
        
        # 拼接原始文档片段，交由 Agent LLM 统一综合回答
        snippets = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            source = doc.metadata.get('source', '未知来源') if hasattr(doc, 'metadata') else '未知来源'
            snippets.append(f"[知识片段{i}] 来源:{source}\n{content}")
            # 打印每个片段的完整内容，方便排查检索质量
            knowledge_logger.info(f'[RAG_CHUNK_{i}] 来源: {source}')
            knowledge_logger.info(f'[RAG_CHUNK_{i}] 内容:\n{content}')
        
        result = (
            "⚠️ 以下是从官方知识库检索到的原始内容，请严格基于这些片段回答，"
            "禁止使用任何片段以外的知识（包括训练数据中的网址、邮箱等），"
            "若片段中找不到所需信息则如实告知用户：\n\n"
        ) + "\n\n".join(snippets)
        knowledge_logger.info(f'[RAG_TOOL] RAG检索成功 | query: {query} | 片段数: {len(docs)}')
        return result
    except Exception as e:
        error_msg = f"RAG查询失败: {str(e)}"
        knowledge_logger.error(f'[RAG_TOOL_ERROR] {error_msg}')
        return f"知识库检索失败：{error_msg}"
