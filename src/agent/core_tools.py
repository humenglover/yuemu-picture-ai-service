from langchain_core.tools import tool

@tool
def think(reasoning: str) -> str:
    """当你认为不需要其他工具时，用这个函数分析用户意图并解释原因。这能让你直接与用户进行对话。"""
    return "分析完成，现在你可以直接输出回答了。"

@tool
def noop() -> str:
    """当你确定没有任何工具能帮助用户时，调用此函数。调用后你会收到消息 '请重新询问用户想要什么'"""
    return "请重新询问用户想要什么"

from langchain_core.runnables import RunnableConfig

@tool
def get_session_history(config: RunnableConfig, limit: int = 20) -> str:
    """当用户询问“以前说了啥”、“上次怎么说的”、“历史记录”等明确要求回顾聊天历史时调用此工具。
可以获取当前会话过去指定轮数的原始对话记录。
参数说明：
- limit (可选): 需要获取的对话轮数，默认为 20 轮（一轮包含一问一答）。
"""
    import json
    import requests
    from utils.log_utils import knowledge_logger
    
    token = config.get("configurable", {}).get("sa_token")
    session_id = config.get("configurable", {}).get("session_id")
    
    if not token or not session_id:
        return '{"error": "未提供身份凭证或会话ID，无法获取历史记录。"}'
    
    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"
        
    history_url = f"{java_base_url}/rag/qa/message/list?sessionId={session_id}&current=1&pageSize={limit * 2}"
    
    headers = {
        "satoken": token
    }
    
    try:
        knowledge_logger.info(f'[AGENT_HISTORY_TOOL] 请求获取 {limit} 轮历史对话')
        response = requests.get(history_url, headers=headers, timeout=10)
        res_data = response.json()
        
        if res_data.get("code") == 0 and res_data.get("data"):
            records = res_data["data"].get("records", [])
            if not records:
                return '{"result": "当前会话没有更多历史记录。"}'
            
            # 由于Java层的倒序逻辑已经在 controller 中处理好（最老的消息在前，最新的在后），直接拼接即可
            # 如果 Java 接口返回的是按照时间升序（旧的在前），那符合人类阅读习惯。
            formatted_history = []
            for msg in records:
                role = "用户" if msg.get("messageType") == 1 else "AI"
                content = msg.get("content", "")
                formatted_history.append(f"[{role}]: {content}")
            
            return json.dumps({
                "status": "success",
                "fetched_rounds": len(records) // 2,
                "history": "\n".join(formatted_history)
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "error": res_data.get("message", "获取历史记录失败")
            }, ensure_ascii=False)
    except Exception as e:
        knowledge_logger.error(f'[AGENT_HISTORY_TOOL_ERROR] 请求异常: {str(e)}')
        return json.dumps({"error": f"调用接口异常: {str(e)}"}, ensure_ascii=False)

@tool
def search_long_term_memory(config: RunnableConfig, keyword: str) -> str:
    """当用户明确询问跨越时间很久的事件、历史总结、或者在当前会话记录中找不到的早期交互时调用。
该工具通过关键词在长期记忆库（历史会话摘要）中进行语义搜索。
参数说明：
- keyword: 搜索关键词或简短描述，例如“滤镜”、“风景照片”、“上个月”等。
"""
    import json
    import requests
    from utils.log_utils import knowledge_logger
    
    token = config.get("configurable", {}).get("sa_token")
    if not token:
        return '{"error": "未提供身份凭证，无法获取记忆。"}'
    
    try:
        from model.factory import load_config
        config_data = load_config()
        java_base_url = config_data.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"
        
    search_url = f"{java_base_url}/rag/memory/search"
    headers = {"satoken": token}
    params = {"keyword": keyword}
    
    try:
        knowledge_logger.info(f'[AGENT_LTM_TOOL] 请求搜索长期记忆 | 关键词: {keyword}')
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 404:
            return json.dumps({"error": "严重系统错误：Java服务尚未重启，找不到 memory/search 接口。请立刻停止重试，直接告诉用户去重启Java服务端！"}, ensure_ascii=False)
            
        res_data = response.json()
        
        if res_data.get("code") == 0 and res_data.get("data"):
            memory_data = res_data["data"]
            if "未检索到相关历史记忆" in str(memory_data):
                return json.dumps({
                    "status": "success",
                    "memory": "未检索到相关历史记忆。指令：请勿再使用其他关键词重试，直接告诉用户没有找到相关记录。"
                }, ensure_ascii=False)
            return json.dumps({
                "status": "success",
                "memory": memory_data
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "error": res_data.get("message", "搜索记忆失败") + "。指令：请勿再尝试，直接告知用户失败。"
            }, ensure_ascii=False)
    except Exception as e:
        knowledge_logger.error(f'[AGENT_LTM_TOOL_ERROR] 请求异常: {str(e)}')
        return json.dumps({"error": f"调用接口异常: {str(e)}。指令：后台异常，请立即停止重试，并告知用户！"}, ensure_ascii=False)
