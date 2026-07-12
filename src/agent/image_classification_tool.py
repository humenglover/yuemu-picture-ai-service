import os
import json
import base64
import requests
import numpy as np
import dashscope
import tempfile
from typing import List, Dict, Optional
from dashscope import MultiModalEmbedding
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from utils.log_utils import knowledge_logger

# 默认类别列表
DEFAULT_CATEGORIES = [
    "风光", "人像", "动物", "植物", "山水", "森林", "海边", "沙漠", "冰雪", "田园", "古建", "地标", "小屋", "工业", "桥梁", "花卉", "绿植", "多肉", "枝叶", "向日葵", "猫咪", "狗狗", "野物", "海洋", "鸟类", "昆虫", "肖像", "情侣", "亲子", "闺蜜", "古风", "街拍", "儿童", "美食", "甜品", "水果", "咖啡", "家常菜", "零食", "静物", "复古", "饰品", "数码", "文具", "玻璃", "合成", "黑白", "渐变", "抽象", "壁纸", "山川", "河流", "湖泊", "草原", "云海", "日出", "日落", "雪景", "高山", "峡谷", "瀑布", "溪流", "自然风景", "城市风光", "乡村风景", "古镇", "寺庙", "园林", "街景", "夜景", "大桥", "高楼", "男性人像", "女性人像", "儿童人像", "老人人像", "个人写真", "艺术照", "生活照", "职业照", "情绪人像", "清新人像", "复古人像", "时尚人像", "宠物猫", "宠物狗", "野生动物", "水中动物", "飞禽", "走兽", "萌宠", "水族", "猛兽", "树木", "花草", "盆栽", "鲜花", "绿叶", "灌木", "藤蔓", "水生植物", "四季植物", "美食摄影", "甜品蛋糕", "饮品", "中餐", "西餐", "水果特写", "静物摄影", "艺术插画", "卡通形象", "动漫", "手绘", "数字艺术"
]

class ImageClassifier:
    def __init__(self, categories: List[str] = DEFAULT_CATEGORIES):
        from model.factory import load_config
        config = load_config()
        self.api_key = config.get("qwen_api_key")
        # 更换为公开可用的 Plus 模型
        self.model = config.get("multimodal_embedding_model_name", "tongyi-embedding-vision-plus-2026-03-06")
        
        # 使用统一的 qdrant 配置加载
        try:
            from model.factory import load_qdrant_config
            qdrant_config = load_qdrant_config()
            qdrant_url = qdrant_config.get('qdrant_url', 'http://localhost:6333')
        except Exception:
            qdrant_url = "http://localhost:6333"
        
        # 多模态嵌入包装类
        class QwenMultimodalEmbeddingFunction:
            def __init__(self, api_key, model):
                self.api_key = api_key
                self.model = model
            def __call__(self, input):
                # 嵌入函数可能传入文档或列表
                return self.embed_documents(input)
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                dashscope.api_key = self.api_key
                embeddings = []
                for text in texts:
                    try:
                        # 官方推荐格式：将 text 放在 content 列表中
                        res = MultiModalEmbedding.call(
                            model=self.model,
                            input=[{'text': text}]
                        )
                        if res.status_code == 200:
                            # 适配新版 API 结构: res.output['embeddings'][0]['embedding']
                            output = res.output
                            if 'embeddings' in output and len(output['embeddings']) > 0:
                                embeddings.append(output['embeddings'][0]['embedding'])
                            elif 'embedding' in output:
                                embeddings.append(output['embedding'])
                            else:
                                knowledge_logger.error(f"[Classifier] 响应中缺少向量数据: {output}")
                                embeddings.append([0.0] * 1152)
                        else:
                            knowledge_logger.error(f"[Classifier] 文本嵌入失败 '{text}': {res.message}")
                            embeddings.append([0.0] * 1152) 
                    except Exception as e:
                        knowledge_logger.error(f"[Classifier] API异常 '{text}': {e}")
                        embeddings.append([0.0] * 1152)
                return embeddings
            def embed_query(self, text: str) -> List[float]:
                return self.embed_documents([text])[0]

        self.qdrant_client = QdrantClient(url=qdrant_url)
        collection_name = "image_tags_v2"


        self.tag_vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=QwenMultimodalEmbeddingFunction(self.api_key, self.model),
        )
        
        self.categories = categories
        self._sync_categories()

    def _sync_categories(self):
        """同步分类向量库"""
        try:
            # 尝试通过 qdrant client 获取数量
            count = self.qdrant_client.count("image_tags_v2").count
        except Exception:
            count = 0

        try:
            if count == 0:
                knowledge_logger.info(f"[Classifier] 正在初始化分类集 (使用模型: {self.model})...")
                # 批量添加以提高效率
                self.tag_vectorstore.add_texts(
                    texts=self.categories,
                    metadatas=[{"name": cat} for cat in self.categories],
                    ids=[f"t_{i}" for i in range(len(self.categories))]
                )
                knowledge_logger.info("[Classifier] 初始化完成")
        except Exception as e:
            knowledge_logger.error(f"[Classifier] 初始化向量库异常: {e}")

    def _download_and_convert(self, image_url: str) -> str:
        """下载图片并保存为临时文件，返回本地 file:// 路径"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # 创建临时文件
            fd, path = tempfile.mkstemp(suffix=".jpg")
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(response.content)
            
            # 返回符合 DashScope 规范的本地路径格式
            return f"file://{path}"
        except Exception as e:
            knowledge_logger.error(f"[Classifier] 绕过防盗链下载失败: {e}")
            raise Exception(f"图片下载失败，请检查 URL 是否有效: {str(e)}")

    def classify(self, image_url: str, top_k: int = 3) -> List[Dict]:
        dashscope.api_key = self.api_key
        tmp_local_path = None
        
        try:
            # 1. 穿透防盗链：下载到本地
            local_url = self._download_and_convert(image_url)
            tmp_local_path = local_url.replace("file://", "")
            
            # 2. 获取图片嵌入向量
            res = MultiModalEmbedding.call(
                model=self.model,
                input=[{'image': local_url}]
            )
            
            if res.status_code != 200:
                raise Exception(f"模型调用失败: {res.message}")
            
            output = res.output
            if 'embeddings' in output and len(output['embeddings']) > 0:
                img_vector = output['embeddings'][0]['embedding']
            elif 'embedding' in output:
                img_vector = output['embedding']
            else:
                raise Exception(f"响应中缺少图片向量数据: {output}")
            
            # 3. 向量搜索匹配分类
            search_results = self.tag_vectorstore.similarity_search_with_score_by_vector(
                embedding=img_vector,
                k=top_k
            )
            
            # 4. 组装结果
            return [
                {
                    "category": doc.page_content,
                    "score": round(float(1 / (1 + score)), 4) # 简单的 L2 转换相似度
                } for doc, score in search_results
            ]
            
        finally:
            # 清理临时文件
            if tmp_local_path and os.path.exists(tmp_local_path):
                os.remove(tmp_local_path)

# 单例管理
_instance = None

@tool(description="对在线图片进行语义分类，能够识别风光、人像、美食等 100+ 场景。")
def classify_image_tool(image_url: str) -> str:
    """输入图片 URL，输出分类结果。已集成防盗链破解与向量加速。"""
    global _instance
    try:
        if _instance is None:
            _instance = ImageClassifier()
        
        res = _instance.classify(image_url)
        return json.dumps({"type": "classification_result", "data": res}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
