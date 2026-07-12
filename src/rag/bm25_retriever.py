"""
BM25 关键词检索器
用于精确匹配术语、编码、专业词汇
"""
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import jieba
from langchain_core.documents import Document
from utils.log_utils import rag_logger


class BM25Retriever:
    """BM25 关键词检索器（支持中文分词）"""
    
    def __init__(self, documents: List[Document] = None):
        """
        初始化 BM25 检索器
        
        Args:
            documents: 文档列表
        """
        self.documents = documents or []
        self.bm25 = None
        self.tokenized_corpus = []
        
        if self.documents:
            self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词（保留英文、数字、特殊符号）
        
        Args:
            text: 待分词文本
            
        Returns:
            分词结果列表
        """
        # 使用 jieba 分词，保留英文和数字
        tokens = jieba.lcut_for_search(text)
        # 过滤空白符
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens
    
    def _build_index(self):
        """构建 BM25 索引"""
        if not self.documents:
            self.bm25 = None
            self.tokenized_corpus = []
            return
        try:
            rag_logger.info(f"[BM25_BUILD_INDEX] 开始构建 BM25 索引 | 文档数量: {len(self.documents)}")
            
            # 对所有文档进行分词
            self.tokenized_corpus = []
            for doc in self.documents:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                tokens = self._tokenize(content)
                self.tokenized_corpus.append(tokens)
            
            # 构建 BM25 索引
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            rag_logger.info(f"[BM25_BUILD_INDEX] BM25 索引构建完成 | 文档数量: {len(self.documents)}")
            
        except Exception as e:
            rag_logger.error(f"[BM25_BUILD_INDEX_ERROR] BM25 索引构建失败: {str(e)}")
            raise
    
    def update_documents(self, documents: List[Document]):
        """
        更新文档并重建索引
        
        Args:
            documents: 新的文档列表
        """
        if not documents:
            return
        self.documents = documents
        self._build_index()
    
    def retrieve(self, query: str, top_k: int = 20) -> List[tuple]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            
        Returns:
            [(doc, score), ...] 文档和分数的元组列表
        """
        if not self.bm25 or not self.documents:
            rag_logger.warning("[BM25_RETRIEVE] BM25 索引未初始化或文档为空")
            return []
        
        try:
            # 对查询进行分词
            query_tokens = self._tokenize(query)
            
            rag_logger.info(f"[BM25_RETRIEVE] 开始 BM25 检索 | query: {query[:50]}... | query_tokens: {len(query_tokens)} | top_k: {top_k}")
            
            # 获取 BM25 分数
            scores = self.bm25.get_scores(query_tokens)
            
            # 获取 top_k 索引
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            # 构建结果
            results = [(self.documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]
            
            rag_logger.info(f"[BM25_RETRIEVE] BM25 检索完成 | 召回数量: {len(results)}")
            
            return results
            
        except Exception as e:
            rag_logger.error(f"[BM25_RETRIEVE_ERROR] BM25 检索失败: {str(e)}")
            return []
