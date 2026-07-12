"""
RRF (Reciprocal Rank Fusion) 融合算法
用于融合向量检索和 BM25 检索的结果
"""
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from utils.log_utils import rag_logger


class RRFFusion:
    """RRF 倒数排序融合"""
    
    def __init__(self, k: int = 60):
        """
        初始化 RRF 融合器
        
        Args:
            k: RRF 常数，默认 60（论文推荐值）
        """
        self.k = k
    
    def fuse(self, 
             vector_results: List[Tuple[Document, float]], 
             bm25_results: List[Tuple[Document, float]], 
             vector_weight: float = 0.5,
             bm25_weight: float = 0.5) -> List[Tuple[Document, float]]:
        """
        RRF 融合算法
        
        公式: score = weight_vector / (k + rank_vector) + weight_bm25 / (k + rank_bm25)
        
        Args:
            vector_results: 向量检索结果 [(doc, score), ...]
            bm25_results: BM25 检索结果 [(doc, score), ...]
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重
            
        Returns:
            融合后的结果 [(doc, fused_score), ...]
        """
        try:
            rag_logger.info(f"[RRF_FUSION] 开始 RRF 融合 | 向量结果: {len(vector_results)} | BM25结果: {len(bm25_results)}")
            
            # 用于存储每个文档的融合分数
            doc_scores = {}
            doc_objects = {}
            
            # 处理向量检索结果
            for rank, (doc, score) in enumerate(vector_results, start=1):
                doc_id = self._get_doc_id(doc)
                rrf_score = vector_weight / (self.k + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
                doc_objects[doc_id] = doc
            
            # 处理 BM25 检索结果
            for rank, (doc, score) in enumerate(bm25_results, start=1):
                doc_id = self._get_doc_id(doc)
                rrf_score = bm25_weight / (self.k + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
                doc_objects[doc_id] = doc
            
            # 按融合分数排序
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 构建最终结果
            fused_results = [(doc_objects[doc_id], score) for doc_id, score in sorted_docs]
            
            rag_logger.info(f"[RRF_FUSION] RRF 融合完成 | 融合后数量: {len(fused_results)}")
            
            return fused_results
            
        except Exception as e:
            rag_logger.error(f"[RRF_FUSION_ERROR] RRF 融合失败: {str(e)}")
            # 降级：返回向量检索结果
            return vector_results
    
    def _get_doc_id(self, doc: Document) -> str:
        """
        获取文档唯一标识
        
        注意：不能使用 source 作为 ID，因为同一个 source 会被切分成多个 Chunk，
        使用 source 会导致不同 Chunk 互相覆盖，丢失上下文信息。
        
        Args:
            doc: 文档对象
            
        Returns:
            文档 ID（每个 Chunk 唯一）
        """
        # 优先使用 metadata 中的 id（应该是 chunk 级别的唯一 ID）
        if hasattr(doc, 'metadata') and 'id' in doc.metadata:
            return str(doc.metadata['id'])
        
        # 使用内容的哈希作为 ID（确保每个 Chunk 都有唯一标识）
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        
        # 如果有 source 和 chunk_id，组合使用以提高可读性
        if hasattr(doc, 'metadata'):
            source = doc.metadata.get('source', '')
            chunk_id = doc.metadata.get('chunk_id', '')
            if source and chunk_id:
                return f"{source}::{chunk_id}"
        
        # 最终降级：使用内容哈希
        return str(hash(content))
