from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class RAGRequestDTO(BaseModel):
    """
    RAG请求数据传输对象
    """
    history: Optional[List[Dict[str, str]]] = []
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 8
    temperature: Optional[float] = 0.7
    long_term_memory: Optional[str] = ""
    sa_token: Optional[str] = None
    user_persona: Optional[str] = ""
    model: Optional[str] = None


class StreamRAGRequestDTO(BaseModel):
    """
    RAG流式请求数据传输对象
    """
    history: Optional[List[Dict[str, str]]] = []
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 8
    temperature: Optional[float] = 0.7
    long_term_memory: Optional[str] = ""
    sa_token: Optional[str] = None
    user_persona: Optional[str] = ""
    model: Optional[str] = None


class KnowledgeUploadRequestDTO(BaseModel):
    """
    知识库上传请求数据传输对象
    """
    filename: str
    content_type: str
    size: int

class AIPostRequestDTO(BaseModel):
    """
    AI 自动生成帖子请求
    """
    prompt: str
    category: Optional[str] = "默认分类"
    style_id: Optional[int] = None
    sa_token: Optional[str] = None

class AIPictureRequestDTO(BaseModel):
    """
    AI 图片配文请求
    """
    image_url: str
    sa_token: Optional[str] = None

class PureLLMChatRequest(BaseModel):
    """
    纯LLM调用请求数据传输对象（不含任何RAG、上下文或元数据）
    """
    prompt: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None