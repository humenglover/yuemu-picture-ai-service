import os
from typing import List, Tuple
import yaml
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, FilterSelector, Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
import traceback

# 路径导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.factory import create_embedding_model
from utils.log_utils import knowledge_logger
from utils.file_utils import (
    calculate_md5,
    check_md5_exists_only,
    check_md5_exists_permanently,
    save_md5_with_filename,
    save_md5_permanently,
    get_file_documents
)


class VectorStoreManager:
    def __init__(self):
        self.config = self.load_qdrant_config()
        self.embedding_model = create_embedding_model()

        # 路径初始化
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.data_path = os.path.join(project_root, self.config.get('data_path', './knowledge').lstrip('./'))

        # 初始化向量库客户端和分词器
        self.qdrant_client = QdrantClient(url=self.config.get('qdrant_url', 'http://localhost:6333'))
        collection_name = self.config['collection_name']

        # text-embedding-v3（阿里云）固定输出维度为 1024，提前建好集合防止首次查询 404
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            knowledge_logger.info(f'[QDRANT_INIT] 已创建新集合: {collection_name}')

        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embedding_model,
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            separators=self.config['separators'],
            length_function=len,
        )

        # 确保目录存在
        os.makedirs(self.data_path, exist_ok=True)

        # 加载知识库
        self.load_knowledge_base()

    def load_qdrant_config(self):
        from model.factory import load_qdrant_config
        return load_qdrant_config()

    def get_retriever(self):
        return self.vectorstore.as_retriever(search_kwargs={"k": self.config['k']})

    def calculate_md5(self, file_path: str) -> str:
        return calculate_md5(file_path)

    def load_knowledge_base(self):
        data_path = self.data_path
        md5_hex_store_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          self.config['md5_hex_store'].lstrip('./'))
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(os.path.dirname(md5_hex_store_path), exist_ok=True)

        # 确保MD5记录文件存在
        if not os.path.exists(md5_hex_store_path):
            with open(md5_hex_store_path, 'w', encoding='utf-8') as f:
                pass

        # 筛选允许的文件
        allowed_files = []
        allowed_exts = self.config['allow_knowledge_file_type']
        try:
            for file_name in os.listdir(data_path):
                file_path = os.path.join(data_path, file_name)
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(file_name)[1].lower()
                    if file_ext in allowed_exts:
                        allowed_files.append(file_path)
        except FileNotFoundError:
            knowledge_logger.warning('[知识库加载] 目录不存在，已自动创建：' + data_path)
            os.makedirs(data_path, exist_ok=True)

        # 日志输出扫描结果
        file_count = str(len(allowed_files))
        knowledge_logger.info(
            '[KNOWLEDGE_BASE_SCAN] 开始扫描知识库目录：' + data_path + ' | 找到可处理文件数：' + file_count)

        for path in allowed_files:
            knowledge_logger.info('[FILE_PROCESS_START] 开始处理文件：' + path)
            md5_hex = self.calculate_md5(path)

            # 文件已存在则跳过
            if check_md5_exists_only(md5_hex, md5_hex_store_path):
                knowledge_logger.info('[FILE_DUPLICATE_SKIP] 文件已存在，跳过：' + path + ' | MD5：' + md5_hex)
                continue

            try:
                # 加载文档
                documents: List[Document] = get_file_documents(path)
                if not documents:
                    continue

                # 分片文档
                split_document: List[Document] = self.text_splitter.split_documents(documents)
                if not split_document:
                    continue

                # 修改文档元数据以包含MD5信息
                for doc in split_document:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['file_md5'] = md5_hex
                    doc.metadata['source'] = path

                # 存入向量库
                self.vectorstore.add_documents(split_document)
                
                # 保存MD5记录
                save_md5_with_filename(md5_hex, os.path.basename(path), md5_hex_store_path)

            except Exception as e:
                err_info = '[FILE_PROCESS_ERROR] 文件处理失败：' + path + ' | 错误信息：' + str(e)
                knowledge_logger.error(err_info)
                traceback.print_exc()
                continue

        # 最终知识库加载完成日志
        try:
            total_docs = str(self.qdrant_client.count(self.config['collection_name']).count)
            knowledge_logger.info('[KNOWLEDGE_BASE_LOADED] 知识库加载完成 | 向量库总片段数：' + total_docs)
        except:
            pass

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        if k is None:
            k = self.config['k']
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        if k is None:
            k = self.config['k']
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def clear_all_vectors(self):
        try:
            collection_name = self.config['collection_name']
            if self.qdrant_client.collection_exists(collection_name):
                # 删除集合并重新创建是最快的清空方式
                self.qdrant_client.delete_collection(collection_name)
                knowledge_logger.info('[VECTOR_STORE_CLEAR] 成功清空向量库集合: ' + collection_name)
            return True
        except Exception as e:
            err_info = str(e)
            knowledge_logger.error('[VECTOR_STORE_CLEAR_ERROR] 清空向量库失败：' + err_info)
            return False

    def clear_file_vectors(self, file_md5: str):
        return self.clear_vectors_by_md5(file_md5)

    def clear_vectors_by_md5(self, file_md5: str):
        try:
            collection_name = self.config['collection_name']
            if self.qdrant_client.collection_exists(collection_name):
                self.qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="metadata.file_md5",
                                    match=MatchValue(value=file_md5)
                                )
                            ]
                        )
                    )
                )
                knowledge_logger.info('[VECTOR_STORE_CLEAR_MD5] 成功清理MD5向量数据 | file_md5: ' + file_md5)
            return True
        except Exception as e:
            err_info = str(e)
            knowledge_logger.error(
                '[VECTOR_STORE_CLEAR_MD5_ERROR] 清理MD5向量数据失败 | file_md5: ' + file_md5 + ' | error: ' + err_info)
            return False