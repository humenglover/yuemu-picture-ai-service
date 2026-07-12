import os
import requests
import json
from typing import Optional, Union
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger
from .context import get_sa_token
from utils.image_utils import analyze_local_image_attributes



@tool(description="""将本地图片文件上传到服务器云存储。
参数说明：
- image_path (必须): 本地图片文件的路径（支持相对路径和绝对路径）
- space_id (可选): 目标空间ID，None或0表示公共空间，-1表示帖子图片
- name (可选): 图片名称，如果不提供则使用文件名
- introduction (可选): 图片简介/描述

重要提示：
1. 此工具用于上传本地图片文件到云服务器
2. 上传后会自动进行图片审核
3. 本工具会在上传前**强制自动在本地提取真实的宽高比(picScale)、宽度(picWidth)、高度(picHeight)和主色调(picColor)**，以确保图片无论是上传到公共空间还是私有空间，前端瀑布流都能得到完美的展示。
4. 返回核心数据：url、thumbnailUrl、name、introduction、tags、category、picSize、picWidth、picHeight、picScale、picFormat、picColor
5. 普通用户上传到公共空间会进入草稿箱，管理员上传会自动通过
6. space_id=-1 表示上传为帖子图片，不会进入图库
7. 支持的图片格式：jpg, jpeg, png, webp, gif等常见格式
""")
def upload_local_image(
    image_path: str,
    space_id: Optional[Union[int, str]] = None,
    name: Optional[str] = None,
    introduction: Optional[str] = None
) -> str:
    """将本地图片文件上传到服务器云存储。需要 sa-token 鉴权。"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)，无法执行上传操作。"}'
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        return f'{{"error": "图片文件不存在: {image_path}"}}'
    
    # 检查是否是文件
    if not os.path.isfile(image_path):
        return f'{{"error": "指定的路径不是文件: {image_path}"}}'
    
    # 处理 space_id
    if isinstance(space_id, str):
        if space_id.lower() in ('none', 'null', 'undefined', ''):
            space_id = None
        elif space_id.lstrip('-').isdigit():
            space_id = int(space_id)
        else:
            space_id = None
    
    # 如果没有提供名称，使用文件名（不含扩展名）
    if not name:
        name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 从配置文件读取Java后端基础路径
    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"
    
    upload_url = f"{java_base_url}/picture/upload/postimage"
    
    # 构建请求头（不包含 Content-Type，让 requests 自动设置）
    headers = {
        "satoken": token
    }
    
    # 构建 multipart/form-data 请求
    try:
        # 打开文件
        with open(image_path, 'rb') as f:
            # 构建 files 参数
            files = {
                'file': (os.path.basename(image_path), f, 'image/*')
            }
            
            # 构建 data 参数（query params）
            params = {}
            if space_id is not None:
                params['spaceId'] = space_id
            if name:
                params['picName'] = name
            if introduction:
                params['introduction'] = introduction
                
            # 提取宽高、比例及主色调，确保在上传前补齐数据并上传
            attrs = analyze_local_image_attributes(image_path)
            if attrs:
                params['picWidth'] = attrs['picWidth']
                params['picHeight'] = attrs['picHeight']
                params['picScale'] = attrs['picScale']
                params['picColor'] = attrs['picColor']
                knowledge_logger.info(f'[IMAGE_UPLOAD_TOOL] 成功提取并上送当前图片天然属性: {json.dumps(attrs)}')
            
            knowledge_logger.info(f'[IMAGE_UPLOAD_TOOL] 开始上传图片 | path: {image_path} | space: {space_id} | name: {name}')
            knowledge_logger.info(f'[IMAGE_UPLOAD_TOOL] 请求参数: {json.dumps(params, ensure_ascii=False)}')
            
            # 发送请求
            response = requests.post(
                upload_url,
                headers=headers,
                params=params,
                files=files,
                timeout=60  # 上传文件可能需要更长时间
            )
            
            result = response.json()
            
            if response.status_code == 200 and result.get("code") == 0:
                data = result.get("data", {})
                pic_id = data.get("id")
                pic_url = data.get("url")
                thumbnail_url = data.get("thumbnailUrl")
                review_status = data.get("reviewStatus", 0)
                review_message = data.get("reviewMessage", "")
                is_draft = data.get("isDraft", 0)
                
                knowledge_logger.info(f'[IMAGE_UPLOAD_TOOL] 图片上传成功 | id: {pic_id} | url: {pic_url}')
                knowledge_logger.info(f'[IMAGE_UPLOAD_TOOL] 审核状态: {review_status} | 审核信息: {review_message}')
                
                # 构建返回消息
                if space_id == -1:
                    msg = "图片已成功上传为帖子图片"
                elif space_id is None or space_id == 0:
                    if is_draft == 1:
                        msg = "图片已保存到草稿箱，您可以在个人中心的草稿箱中查看并发布"
                    else:
                        msg = "图片已成功上传并自动审核通过，已在公共空间展示"
                else:
                    msg = "图片已成功保存到您的空间"
                
                # 返回核心图片信息（简化版，只包含AI需要知道的字段）
                return json.dumps({
                    "type": "image_uploaded",
                    "msg": msg,
                    "url": pic_url,
                    "thumbnailUrl": thumbnail_url,
                    "name": data.get("name"),
                    "introduction": data.get("introduction"),
                    "tags": data.get("tags", []),
                    "category": data.get("category"),
                    "picSize": data.get("picSize"),
                    "picWidth": data.get("picWidth"),
                    "picHeight": data.get("picHeight"),
                    "picScale": data.get("picScale"),
                    "picFormat": data.get("picFormat"),
                    "picColor": data.get("picColor")
                }, ensure_ascii=False)
            else:
                msg = result.get("message", "未知错误")
                knowledge_logger.error(f'[IMAGE_UPLOAD_TOOL_ERROR] 上传失败: {msg}')
                return f'{{"error": "Java后端返回错误: {msg}"}}'
                
    except FileNotFoundError:
        return f'{{"error": "无法打开图片文件: {image_path}"}}'
    except requests.exceptions.Timeout:
        knowledge_logger.error(f'[IMAGE_UPLOAD_TOOL_ERROR] 上传超时')
        return '{"error": "上传超时，请检查网络连接或稍后重试"}'
    except requests.exceptions.RequestException as e:
        knowledge_logger.error(f'[IMAGE_UPLOAD_TOOL_ERROR] 请求异常: {str(e)}')
        return f'{{"error": "调用上传接口失败: {str(e)}"}}'
    except Exception as e:
        knowledge_logger.error(f'[IMAGE_UPLOAD_TOOL_ERROR] 未知异常: {str(e)}')
        return f'{{"error": "上传过程中发生错误: {str(e)}"}}'
