import os
import io
import uuid
import json
import re
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger
from model.factory import load_config
from .context import get_sa_token

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tts.v20190823 import tts_client, models

# 1. 扩充声线库，赋予 AI 更丰富的情感表现力
# 腾讯云音色ID映射:
VOICE_MAP = {
    "female_gentle": 101001, # 智妍 - 情感女声 (默认)
    "female_lively": 101016, # 智甜 - 甜美女声
    "male_bright": 101015,   # 智明 - 阳光男声
    "male_mature": 101002,   # 智诚 - 成熟男声
    "child": 101011          # 智佳 - 儿童女声
}

def _clean_text_for_tts(text: str) -> str:
    """清洗文本，移除不支持发音的 Markdown 符号，限制长度防止内存爆炸"""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'\n+', '，', text)
    return text[:800]

def _chunk_text(text: str, max_len: int = 140) -> list:
    chunks = []
    import re
    # 按照常见句号断句保留分隔符
    sentences = re.split(r'([。！？\n.!?])', text)
    current_chunk = ""
    for piece in sentences:
        if len(current_chunk) + len(piece) <= max_len:
            current_chunk += piece
        else:
            if current_chunk.strip():
                chunks.append(current_chunk)
            if len(piece) > max_len:
                for i in range(0, len(piece), max_len):
                    if piece[i:i+max_len].strip():
                        chunks.append(piece[i:i+max_len])
                current_chunk = ""
            else:
                current_chunk = piece
    if current_chunk.strip():
        chunks.append(current_chunk)
    return chunks

def _generate_tencent_tts(text: str, voice_type_id: int) -> bytes:
    """底层同步生成腾讯云语音流内容（支持长文本自动切片合并）"""
    config = load_config()
    try:
        secret_id = config.get('tencentcloud', {}).get('secret_id')
        secret_key = config.get('tencentcloud', {}).get('secret_key')
        if not secret_id or not secret_key:
            raise KeyError
    except KeyError:
        raise ValueError("腾讯云 TTS 需要配置 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY 环境变量，请在 .env 文件中设置")

    try:
        cred = credential.Credential(secret_id, secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tts.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile

        client = tts_client.TtsClient(cred, "", clientProfile)

        # 文本过长分段处理 (腾讯云限 150 中文字符)
        chunks = _chunk_text(text, 140)
        final_audio_bytes = b""

        for chunk in chunks:
            if not chunk.strip():
                continue
            req = models.TextToVoiceRequest()
            req.Text = chunk
            req.SessionId = uuid.uuid4().hex
            req.VoiceType = voice_type_id
            req.Codec = "mp3"

            resp = client.TextToVoice(req)
            audio_base64 = resp.Audio
            if audio_base64:
                final_audio_bytes += base64.b64decode(audio_base64)
        
        if not final_audio_bytes:
            raise ValueError("腾讯云 TTS 接口未返回 Audio 数据")
            
        return final_audio_bytes
    except TencentCloudSDKException as err:
        raise Exception(f"TencentCloudSDKException: {str(err)}")
    except Exception as e:
        raise e

def _get_requests_session_with_retries():
    """鲁棒性优化：创建一个带有指数退避重试机制的请求会话，防止网络抖动"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

@tool(description="""将指定的文本转换为语音回复。
当用户明确要求你‘说话’、‘发语音’、‘听听你的声音’，或者你认为需要用更具感染力的声音回答时（例如温馨的提醒、鼓励的话语），请调用此工具。
输入参数:
- text (必须): 需要转换的文字内容。
- voice_type (可选): 声音类型。可选值：'female_gentle'(温柔女声/默认), 'female_lively'(活泼女声), 'male_bright'(阳光男声), 'male_mature'(成熟男声), 'child'(童声)。

注意：该工具会自动调用后端上传接口，并返回一个音频URL。
**重要指令**：调用成功后，你必须在回复给用户的正文末尾单独一行附带 `[语音: 具体的URL]`，以便前端渲染播放器收听。""")
def tts_reply(text: str, voice_type: str = "female_gentle") -> str:
    """AI 免费语音合成并自动上传工具。"""
    try:
        # 预处理清理文本
        clean_text = _clean_text_for_tts(text)
        if not clean_text.strip():
            return '{"error": "需要合成的文本清洗后为空"}'
            
        voice_id = VOICE_MAP.get(voice_type, 101001)
        knowledge_logger.info(f'[TTS_TOOL] 正在生成语音(Tencent TTS VoiceId={voice_id}) | 内容: {clean_text[:20]}...')
        
        token = get_sa_token()
        if not token:
            return '{"error": "未提供身份凭证(sa-token)，无法执行上传操作。"}'

        # 执行音频生成
        try:
            audio_bytes = _generate_tencent_tts(clean_text, voice_id)
        except Exception as api_e:
            knowledge_logger.error(f'[TTS_TOOL_ERROR] 音频生成失败: {str(api_e)}')
            return f'{{"error": "腾讯云 TTS 生成异常: {str(api_e)}"}}'

        if not audio_bytes:
            return '{"error": "生成的音频数据为空"}'

        # 读取配置并组装上传请求
        try:
            config = load_config()
            java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
        except Exception:
            java_base_url = "http://127.0.0.1:8123/api"

        upload_url = f"{java_base_url}/audio/upload"
        headers = {"satoken": token}
        
        files = {
            'file': (f"ai_reply_{uuid.uuid4().hex[:6]}.mp3", io.BytesIO(audio_bytes), 'audio/mpeg')
        }
        
        data = {
            'title': f'AI语音({voice_type})',
            'description': f'AI合成语音: {clean_text[:30]}...'
        }

        # 执行带有重试机制的高鲁棒性网络请求
        knowledge_logger.info(f'[TTS_TOOL] 正在上传音频到 Java 后端 | URL: {upload_url}')
        session = _get_requests_session_with_retries()
        try:
            # timeout=(连接超时，读取超时)
            response = session.post(upload_url, files=files, data=data, headers=headers, timeout=(10, 45))
            response.raise_for_status() # 快速捕获 4xx/5xx 等 HTTP 错误
            result = response.json()
        except requests.exceptions.RequestException as req_e:
            knowledge_logger.error(f'[TTS_TOOL_ERROR] 请求 Java 后端失败/超时: {str(req_e)}')
            return f'{{"error": "云端上传请求失败，网络或服务异常: {str(req_e)}"}}'

        # 解析最终业务结果
        if result.get("code") == 0:
            audio_url = result.get("data", {}).get("fileUrl")
            audio_id = result.get("data", {}).get("id")
            knowledge_logger.info(f'[TTS_TOOL] 语音上传成功 | ID: {audio_id} | URL: {audio_url}')
            
            return json.dumps({
                "type": "audio",
                "url": audio_url,
                "msg": "语音合成成功",
                "hint": f"请务必在你的回答末尾加上这一句（单独一行）：[语音: {audio_url}]"
            }, ensure_ascii=False)
        else:
            msg = result.get("message", "未知错误")
            knowledge_logger.error(f'[TTS_TOOL_ERROR] Java后端业务拒绝: {msg}')
            return f'{{"error": "云端业务处理失败: {msg}"}}'
            
    except Exception as e:
        knowledge_logger.error(f'[TTS_TOOL_ERROR] 工具全链路意外异常: {str(e)}')
        return f'{{"error": "调用语音工具链异常: {str(e)}"}}'
# Trigger hot reload
