from contextvars import ContextVar
from typing import Optional, Dict, Any
from threading import Lock

# 定义存储 sa-token 的上下文变量
sa_token_var: ContextVar[Optional[str]] = ContextVar("sa_token", default=None)

# 【改进】使用全局变量 + 锁来存储图片元数据（更可靠）
_picture_metadata_lock = Lock()
_picture_metadata_global: Optional[Dict[str, Any]] = None

def set_sa_token(token: Optional[str]):
    """设置当前上下文的 sa-token"""
    sa_token_var.set(token)

def get_sa_token() -> Optional[str]:
    """获取当前上下文的 sa-token"""
    return sa_token_var.get()

session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)

def set_session_id(session_id: Optional[str]):
    """设置当前上下文的 session_id"""
    session_id_var.set(session_id)

def get_session_id() -> Optional[str]:
    """获取当前上下文的 session_id"""
    return session_id_var.get()

def set_picture_metadata(metadata: Optional[Dict[str, Any]]):
    """设置当前上下文的图片元数据（使用全局变量）"""
    global _picture_metadata_global
    with _picture_metadata_lock:
        _picture_metadata_global = metadata

def get_picture_metadata() -> Optional[Dict[str, Any]]:
    """获取当前上下文的图片元数据（从全局变量）"""
    global _picture_metadata_global
    with _picture_metadata_lock:
        return _picture_metadata_global

def clear_picture_metadata():
    """清除当前上下文的图片元数据"""
    global _picture_metadata_global
    with _picture_metadata_lock:
        _picture_metadata_global = None
