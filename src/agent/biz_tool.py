import os
import requests
import json
from typing import Optional, Union
from langchain_core.tools import tool
from utils.log_utils import knowledge_logger
from .context import get_sa_token, get_picture_metadata

@tool(description="""将网络图片上传到用户的图库空间。
参数说明：
- file_url (必须): 图片的网络URL地址
- space_id (可选): 目标空间ID，None或0表示公共空间
- name (可选): 图片名称，由用户指定
- introduction (可选): 图片简介/描述，可以由AI根据用户意图生成
- picture_metadata (可选): 图片元数据对象，包含以下字段：
  - thumbnailUrl: 缩略图URL
  - picSize: 图片大小（字节）
  - picWidth: 图片宽度（像素）
  - picHeight: 图片高度（像素）
  - picScale: 图片宽高比
  - picFormat: 图片格式（如webp、jpg等）
  - picColor: 图片主色调（十六进制颜色值）

重要提示：
1. 如果用户没有明确指定图片名称，应该询问用户或根据图片内容智能命名
2. 图片描述(introduction)应该根据用户的上传意图或图片内容生成，不要留空
3. 普通用户上传到公共空间的图片会进入草稿箱，需要用户在个人中心发布
4. 管理员上传到公共空间的图片会自动审核通过，直接公开展示
5. 上传到私有/团队空间的图片无需审核，直接可用
6. 标签和分类由用户后续在管理界面中补充，AI上传时不设置
7. 本工具会在上传前**自动在本地解析图片的真实宽高、比例和主色调**，确保入库信息完整，保障前端瀑布流展示效果。
""")
def upload_picture_to_space(
    file_url: str, 
    space_id: Optional[Union[int, str]] = None, 
    name: Optional[str] = None,
    introduction: Optional[str] = None,
    picture_metadata: Optional[dict] = None
) -> str:
    """将指定的图片URL上传到用户的图库空间。需要 sa-token 鉴权。"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)，无法执行上传操作。"}'
    
    # 【新增】如果没有提供元数据参数，尝试从上下文中获取
    knowledge_logger.info(f'[BIZ_TOOL_DEBUG] 工具调用时 picture_metadata 参数: {picture_metadata}')
    if not picture_metadata:
        ctx_meta = get_picture_metadata()
        if ctx_meta and isinstance(ctx_meta, dict):
            orig_url = ctx_meta.get("url") or ctx_meta.get("fileUrl") or ""
            if orig_url and (orig_url in file_url or file_url in orig_url):
                picture_metadata = ctx_meta
                knowledge_logger.info(f'[BIZ_TOOL_DEBUG] URL匹配，从上下文复用 picture_metadata: {picture_metadata}')
            else:
                knowledge_logger.info('[BIZ_TOOL_DEBUG] URL不匹配(新生成/裁剪图片)，不复用原图上下文元数据')
        
    if isinstance(space_id, str):
        if space_id.lower() in ('none', 'null', 'undefined', ''):
            space_id = None
        elif space_id.isdigit():
            space_id = int(space_id)
        else:
            space_id = None
    
    # 从配置文件读取Java后端基础路径
    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"
        
    upload_url = f"{java_base_url}/picture/upload/url"
    
    headers = {
        "satoken": token,
        "Content-Type": "application/json"
    }
    
    payload = {
        "fileUrl": file_url,
        "spaceId": space_id,
        "picName": name,
        "introduction": introduction
    }
    
    # 【新增】如果提供了图片元数据，添加到请求中
    if picture_metadata and isinstance(picture_metadata, dict):
        if "thumbnailUrl" in picture_metadata:
            payload["thumbnailUrl"] = picture_metadata["thumbnailUrl"]
        if "picSize" in picture_metadata:
            # 确保 picSize 是整数
            try:
                payload["picSize"] = int(picture_metadata["picSize"]) if picture_metadata["picSize"] else None
            except (ValueError, TypeError):
                pass
        if "picWidth" in picture_metadata:
            try:
                payload["picWidth"] = int(picture_metadata["picWidth"]) if picture_metadata["picWidth"] else None
            except (ValueError, TypeError):
                pass
        if "picHeight" in picture_metadata:
            try:
                payload["picHeight"] = int(picture_metadata["picHeight"]) if picture_metadata["picHeight"] else None
            except (ValueError, TypeError):
                pass
        if "picScale" in picture_metadata:
            try:
                payload["picScale"] = float(picture_metadata["picScale"]) if picture_metadata["picScale"] else None
            except (ValueError, TypeError):
                pass
        if "picFormat" in picture_metadata:
            payload["picFormat"] = picture_metadata["picFormat"]
        if "picColor" in picture_metadata:
            payload["picColor"] = picture_metadata["picColor"]
            
    # 【新增核心】如果 picture_metadata 不包含宽高或主色调，则强制下载图片到本地进行特征提取
    # 这样确保无论是传到公共空间、私有空间，都能完美带有宽、高、比例、主色调，实现完美瀑布流。
    if not payload.get("picWidth") or not payload.get("picColor"):
        import tempfile
        from utils.image_utils import analyze_local_image_attributes
        try:
            knowledge_logger.info(f'[BIZ_TOOL] 开始在本地临时下载图片以提取真实属性: {file_url}')
            tmp_resp = requests.get(file_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if tmp_resp.status_code == 200:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(tmp_resp.content)
                    tmp_path = tmp_file.name
                try:
                    attrs = analyze_local_image_attributes(tmp_path)
                    if attrs:
                        payload['picWidth'] = attrs.get('picWidth', payload.get('picWidth'))
                        payload['picHeight'] = attrs.get('picHeight', payload.get('picHeight'))
                        payload['picScale'] = attrs.get('picScale', payload.get('picScale'))
                        payload['picColor'] = attrs.get('picColor', payload.get('picColor'))
                        knowledge_logger.info(f'[BIZ_TOOL] 本地强制提取属性成功: {json.dumps(attrs)}')
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        except Exception as attr_e:
            knowledge_logger.warning(f'[BIZ_TOOL] 尝试本地提取图片属性失败: {str(attr_e)}')
    
    # 移除值为 None 的字段
    payload = {k: v for k, v in payload.items() if v is not None}
    
    try:
        knowledge_logger.info(f'[BIZ_TOOL] 尝试上传图片 | url: {file_url} | space: {space_id} | name: {name} | 有元数据: {picture_metadata is not None}')
        if picture_metadata:
            knowledge_logger.info(f'[BIZ_TOOL] 图片元数据详情: {json.dumps(picture_metadata, ensure_ascii=False)}')
        knowledge_logger.info(f'[BIZ_TOOL] 上传请求payload: {json.dumps(payload, ensure_ascii=False)}')
        
        response = requests.post(upload_url, json=payload, headers=headers, timeout=30)
        result = response.json()
        
        if response.status_code == 200 and result.get("code") == 0:
            pic_id = result.get("data", {}).get("id")
            is_draft = result.get("data", {}).get("isDraft", 0)
            review_status = result.get("data", {}).get("reviewStatus", 0)
            
            knowledge_logger.info(f'[BIZ_TOOL] 图片上传成功 | id: {pic_id} | isDraft: {is_draft} | reviewStatus: {review_status}')
            
            # 根据空间类型和草稿状态返回不同的消息
            if space_id is None or space_id == 0:
                if is_draft == 1:
                    # 普通用户：进入草稿箱
                    msg = "图片已保存到草稿箱，您可以在个人中心的草稿箱中查看并发布"
                else:
                    # 管理员：自动审核通过
                    msg = "图片已成功上传并自动审核通过，已在公共空间展示"
            else:
                msg = "图片已成功保存到您的空间"
            
            return json.dumps({
                "type": "upload_success", 
                "picture_id": pic_id,
                "is_draft": is_draft,
                "msg": msg
            }, ensure_ascii=False)
        else:
            msg = result.get("message", "未知错误")
            knowledge_logger.error(f'[BIZ_TOOL_ERROR] 上传失败: {msg}')
            return f'{{"error": "Java后端返回错误: {msg}"}}'
    except Exception as e:
        knowledge_logger.error(f'[BIZ_TOOL_ERROR] 请求异常: {str(e)}')
        return f'{{"error": "调用上传接口失败: {str(e)}"}}'

@tool(description="查询当前用户所有可用的图库空间列表（包括公共空间、私有空间和加入的团队空间）。返回列表包含每个空间的 ID、名称和类型。")
def list_available_spaces() -> str:
    """获取用户有权访问的空间列表。用于在上传前确定目标 space_id。"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)，无法查询空间。"}'
        
    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"
        
    headers = {
        "satoken": token,
        "Content-Type": "application/json"
    }
    
    available_spaces = []
    # 1. 默认包含公共空间
    available_spaces.append({"id": 0, "name": "公共空间", "type": "公共"})
    
    try:
        knowledge_logger.info('[BIZ_TOOL] 正在查询用户空间列表...')
        
        # 2. 调用后端专门的 AI 接口获取用户有权限的私有和团队空间
        res = requests.get(f"{java_base_url}/space/list/ai", headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json().get("data", [])
            for rec in data:
                space_type = "私有" if rec.get("spaceType") == 0 else "团队"
                available_spaces.append({
                    "id": rec.get("id"),
                    "name": rec.get("spaceName", "未命名空间"),
                    "type": space_type
                })
        
        knowledge_logger.info(f'[BIZ_TOOL] 空间列表查询完成，共找到 {len(available_spaces)} 个空间')
        return JSONUtil_to_str(available_spaces)
        
    except Exception as e:
        knowledge_logger.error(f'[BIZ_TOOL_ERROR] 查询空间异常: {str(e)}')
        return f'{{"error": "查询空间接口失败: {str(e)}", "partial_data": {json.dumps(available_spaces, ensure_ascii=False)}}}'

def JSONUtil_to_str(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)

@tool(description="删除用户自己的图片。必须提供图片的数字 ID（picture_id）才能执行删除，如果用户未提供具体 ID，必须拒绝并要求用户提供。注意：后端会严格校验权限，只能删除用户自己的图片。")
def delete_picture(picture_id: int) -> str:
    """删除指定 ID 的图片。仅能删除当前登录用户自己的图片，权限由后端严格校验。"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)，无法执行删除操作。"}'

    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"

    headers = {
        "satoken": token,
        "Content-Type": "application/json"
    }

    try:
        knowledge_logger.info(f'[BIZ_TOOL] 尝试删除图片 | picture_id: {picture_id}')
        response = requests.post(
            f"{java_base_url}/picture/delete",
            json={"id": picture_id},
            headers=headers,
            timeout=10
        )
        result = response.json()

        if response.status_code == 200 and result.get("code") == 0:
            knowledge_logger.info(f'[BIZ_TOOL] 图片删除成功 | picture_id: {picture_id}')
            return f'{{"type": "delete_success", "picture_id": {picture_id}, "msg": "图片已成功删除"}}'
        else:
            msg = result.get("message", "未知错误")
            knowledge_logger.error(f'[BIZ_TOOL_ERROR] 删除失败: {msg}')
            return f'{{"error": "删除失败: {msg}"}}'
    except Exception as e:
        knowledge_logger.error(f'[BIZ_TOOL_ERROR] 删除请求异常: {str(e)}')
        return f'{{"error": "调用删除接口失败: {str(e)}"}}'

@tool(description="根据关键词联机搜索站内保存的图片资源。凡是用户表现出‘想看图’、‘找图’、‘搜索图片’（包括‘搜索一下’、‘看看有没有’、‘给我几张’等口语化请求）的意图，且目标是具体的物象（如荷花、风景等）时，必须调用此工具。参数说明：- search_text (必须): 搜索关键词。- current (可选): 页码，默认为 1。- page_size (可选): 每页数量，默认为 10。")
def search_pictures(search_text: str, current: int = 1, page_size: int = 10) -> str:
    """在站内语义搜索图片，支持分页。需要 sa-token 鉴权。"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)，无法执行搜索操作。"}'

    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"

    search_url = f"{java_base_url}/picture/search/semantic"

    headers = {
        "satoken": token,
        "Content-Type": "application/json"
    }

    payload = {
        "searchText": search_text,
        "spaceId": 0,
        "current": current,
        "pageSize": page_size
    }

    try:
        knowledge_logger.info(f'[BIZ_TOOL] 尝试语义搜索图片 | keyword: {search_text}')
        response = requests.post(search_url, json=payload, headers=headers, timeout=10)
        result = response.json()

        if response.status_code == 200 and result.get("code") == 0:
            data = result.get("data", {})
            # 语义搜索返回的是 Page<PictureVO>，结果集在 records 中
            records = data.get("records", [])
            total = data.get("total", 0)
            pages = data.get("pages", 0)
            
            # 提取关键信息，避免返回过多无关字段导致 Token 超限
            simplified_records = []
            for rec in records:
                simplified_records.append({
                    "id": rec.get("id"),
                    "name": rec.get("name"),
                    "url": rec.get("url"),
                    "thumbnailUrl": rec.get("thumbnailUrl")
                })
            
            knowledge_logger.info(f'[BIZ_TOOL] 图片搜索成功 | 找到 {len(simplified_records)} 条结果，共 {total} 条')
            return json.dumps({
                "type": "search_success",
                "keyword": search_text,
                "current": current,
                "total": total,
                "pages": pages,
                "count": len(simplified_records),
                "records": simplified_records
            }, ensure_ascii=False)
        else:
            msg = result.get("message", "未知错误")
            knowledge_logger.error(f'[BIZ_TOOL_ERROR] 搜索失败: {msg}')
            return f'{"error": "Java后端返回错误: {msg}"}'
    except Exception as e:
        knowledge_logger.error(f'[BIZ_TOOL_ERROR] 搜索请求异常: {str(e)}')
        return f'{{"error": "调用搜索接口失败: {str(e)}"}}'

@tool(description="""通过给定的图片URL或站内图片ID，在站内全网（公共图库）进行以图搜图（基于Qdrant向量相似度检索）。
当用户提供一张图片或图片链接并要求查找相似图片时，必须调用此工具。
参数说明：
- image_url (可选): 需要搜索的图片网络URL。
- picture_id (可选): 站内已有的图片数字ID。
注意：image_url 和 picture_id 二者必须提供其一。
- current (可选): 页码，默认为 1。
- page_size (可选): 每页数量，默认为 10。
""")
def search_pictures_by_image(image_url: Optional[str] = None, picture_id: Optional[int] = None, current: int = 1, page_size: int = 10) -> str:
    """在站内基于图片进行以图搜图相似度检索，支持分页。需要 sa-token 鉴权。"""
    if not image_url and not picture_id:
        return '{"error": "必须提供 image_url 或 picture_id 其一"}'
        
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)，无法执行搜索操作。"}'

    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"

    search_url = f"{java_base_url}/picture/search/picture"

    headers = {
        "satoken": token,
        "Content-Type": "application/json"
    }

    payload = {
        "spaceId": 0,
        "current": current,
        "pageSize": page_size
    }
    
    if image_url:
        payload["imageUrl"] = image_url
    if picture_id:
        payload["pictureId"] = picture_id

    try:
        knowledge_logger.info(f'[BIZ_TOOL] 尝试以图搜图 | url: {image_url} | id: {picture_id}')
        response = requests.post(search_url, json=payload, headers=headers, timeout=15)
        result = response.json()

        if response.status_code == 200 and result.get("code") == 0:
            data = result.get("data", {})
            records = data.get("records", [])
            total = data.get("total", 0)
            pages = data.get("pages", 0)
            
            # 提取关键信息，避免返回过多无关字段导致 Token 超限
            simplified_records = []
            for rec in records:
                simplified_records.append({
                    "id": rec.get("id"),
                    "name": rec.get("name"),
                    "url": rec.get("url"),
                    "thumbnailUrl": rec.get("thumbnailUrl")
                })
            
            knowledge_logger.info(f'[BIZ_TOOL] 以图搜图成功 | 找到 {len(simplified_records)} 条结果，共 {total} 条')
            return json.dumps({
                "type": "search_by_image_success",
                "current": current,
                "total": total,
                "pages": pages,
                "count": len(simplified_records),
                "records": simplified_records
            }, ensure_ascii=False)
        else:
            msg = result.get("message", "未知错误")
            knowledge_logger.error(f'[BIZ_TOOL_ERROR] 搜索失败: {msg}')
            return f'{{"error": "Java后端返回错误: {msg}"}}'
    except Exception as e:
        knowledge_logger.error(f'[BIZ_TOOL_ERROR] 搜索请求异常: {str(e)}')
        return f'{{"error": "调用搜索接口失败: {str(e)}"}}'

@tool(description="""执行全站综合搜索，能够同时检索出站内的用户、空间和帖子。
当用户表现出查找、搜索、寻找某人、某空间或某帖子的意图时，必须调用此工具。
参数说明：
- query (必须): 检索关键词
- search_type (可选): 限制搜索类型，如果不传则默认依次对3种类型（user, space, post）进行全量搜索并聚合返回。
                      可选值有：'user' (用户), 'space' (团队空间), 'post' (帖子)。
- current (可选): 页码，默认为 1。
- page_size (可选): 每页数量，默认为 5。
""")
def search_site(query: str, search_type: Optional[str] = None, current: int = 1, page_size: int = 5) -> str:
    """全站通用综合搜索，支持用户、空间、帖子。包含直达前端路由。"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)，无法执行全站搜索操作。"}'

    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"

    search_url = f"{java_base_url}/search/all"
    headers = {
        "satoken": token,
        "Content-Type": "application/json"
    }

    types_to_search = []
    if search_type and search_type in ["user", "space", "post"]:
        types_to_search = [search_type]
    else:
        types_to_search = ["user", "space", "post"]

    all_results = {
        "user": [],
        "space": [],
        "post": []
    }

    for t in types_to_search:
        payload = {
            "searchText": query,
            "type": t,
            "current": current,
            "pageSize": page_size
        }
        try:
            knowledge_logger.info(f'[BIZ_TOOL] 全站搜索：正在检索类型 {t} | query: {query}')
            response = requests.post(search_url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    data = result.get("data", {})
                    records = data.get("content", [])
                    
                    # 针对不同类型提取关键字段并注入前端链接
                    for rec in records:
                        rec_id = rec.get("id")
                        if not rec_id:
                            continue
                        
                        if t == "user":
                            all_results["user"].append({
                                "id": rec_id,
                                "userName": rec.get("userName"),
                                "userAccount": rec.get("userAccount"),
                                "userAvatar": rec.get("userAvatar"),
                                "userProfile": rec.get("userProfile"),
                                "route_link": f"/user/{rec_id}",
                                "hash_link": f"/user/{rec_id}"
                            })
                        elif t == "space":
                            all_results["space"].append({
                                "id": rec_id,
                                "spaceName": rec.get("spaceName"),
                                "spaceType": rec.get("spaceType"),
                                "spaceLevel": rec.get("spaceLevel"),
                                "route_link": f"/space/{rec_id}",
                                "hash_link": f"/space/{rec_id}"
                            })
                        elif t == "post":
                            all_results["post"].append({
                                "id": rec_id,
                                "title": rec.get("title"),
                                "coverUrl": rec.get("coverUrl"),
                                "tags": rec.get("tags") or [],
                                "route_link": f"/post/{rec_id}",
                                "hash_link": f"/post/{rec_id}"
                            })
                else:
                    knowledge_logger.warn(f'[BIZ_TOOL_WARNING] 类型 {t} 搜索返回错误: {result.get("message")}')
            else:
                knowledge_logger.error(f'[BIZ_TOOL_ERROR] 类型 {t} 搜索 HTTP 异常: {response.status_code}')
        except Exception as ex:
            knowledge_logger.error(f'[BIZ_TOOL_ERROR] 类型 {t} 搜索异常: {str(ex)}')

    # 汇总输出
    total_count = sum(len(v) for v in all_results.values())
    knowledge_logger.info(f'[BIZ_TOOL] 全站搜索完成，共找到 {total_count} 条结果')
    return json.dumps({
        "type": "site_search_success",
        "query": query,
        "total_results": total_count,
        "results": all_results
    }, ensure_ascii=False)


@tool(description="""查询当前登录用户的个人基本资料、粉丝关注数以及其个人活动历史数据（如自己发布的图片、帖子、点赞、收藏、分享、评论的历史记录以及草稿箱）。
参数说明：
- data_type (必须): 要查询的数据类型，可选值有：
  - 'profile': 获取当前用户的基本资料（包含用户名、账号、头像、简介、性别、生日、地区、关注数和粉丝数）
  - 'picture': 获取当前用户自己发布的图片列表（默认按创建时间倒序）
  - 'post': 获取当前用户自己发布的帖子列表
  - 'likes': 获取当前用户的点赞历史记录
  - 'favorites': 获取当前用户的收藏历史记录
  - 'shares': 获取当前用户的分享历史记录
  - 'comments': 获取当前用户的评论历史记录
  - 'drafts': 获取当前用户的图片草稿列表
- current (可选): 分页查询的当前页码，默认为 1
- page_size (可选): 分页查询的每页大小，默认为 10
""")
def get_my_personal_data(
    data_type: str,
    current: Optional[int] = 1,
    page_size: Optional[int] = 10
) -> str:
    """获取当前登录用户的个人基本资料和各种活动历史记录。需要 sa-token 鉴权。"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供身份凭证(sa-token)，无法执行查询当前用户个人数据操作。"}'

    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"

    headers = {
        "satoken": token,
        "Content-Type": "application/json"
    }

    try:
        knowledge_logger.info(f'[BIZ_TOOL] 正在查询个人数据 | 类型: {data_type} | 页码: {current} | 每页大小: {page_size}')

        # 1. 首先通过 /user/get/login 拿到当前登录用户的信息，用来获取用户 ID 供后续分页接口使用
        login_res = requests.get(f"{java_base_url}/user/get/login", headers=headers, timeout=10)
        if login_res.status_code != 200:
            return f'{{"error": "获取当前登录用户失败，HTTP 状态码: {login_res.status_code}"}}'
        
        login_data = login_res.json()
        if login_data.get("code") != 0:
            return f'{{"error": "获取当前登录用户接口报错: {login_data.get("message")}"}}'
            
        user_info = login_data.get("data", {})
        user_id = user_info.get("id")
        if not user_id:
            return '{"error": "未获取到当前登录用户的有效 ID"}'

        # 2. 根据不同的数据类型进行分别处理
        if data_type == 'profile':
            # 查询关注数和粉丝数
            follow_count = 0
            fans_count = 0
            try:
                follow_res = requests.post(
                    f"{java_base_url}/userfollows/getfollowandfanscount/{user_id}",
                    headers=headers,
                    timeout=5
                )
                if follow_res.status_code == 200:
                    follow_data = follow_res.json().get("data", {})
                    follow_count = follow_data.get("followCount", 0)
                    fans_count = follow_data.get("fansCount", 0)
            except Exception as fe:
                knowledge_logger.warning(f'[BIZ_TOOL] 查询关注粉丝数失败: {str(fe)}')

            profile_result = {
                "id": user_id,
                "userAccount": user_info.get("userAccount"),
                "userName": user_info.get("userName"),
                "userAvatar": user_info.get("userAvatar"),
                "userProfile": user_info.get("userProfile"),
                "gender": user_info.get("gender"),
                "birthday": user_info.get("birthday"),
                "region": user_info.get("region"),
                "personalSign": user_info.get("personalSign"),
                "userTags": user_info.get("userTags"),
                "userRole": user_info.get("userRole"),
                "createTime": user_info.get("createTime"),
                "followCount": follow_count,
                "fansCount": fans_count
            }
            return json.dumps({"type": "profile_success", "data": profile_result}, ensure_ascii=False)

        elif data_type == 'drafts':
            # 查询图片草稿箱
            draft_res = requests.get(f"{java_base_url}/picture/draft/list", headers=headers, timeout=10)
            if draft_res.status_code == 200:
                result_data = draft_res.json().get("data", [])
                # 简化字段，避免 tokens 超载
                simplified_drafts = []
                for item in result_data:
                    simplified_drafts.append({
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "url": item.get("url"),
                        "thumbnailUrl": item.get("thumbnailUrl"),
                        "category": item.get("category"),
                        "tags": item.get("tags"),
                        "editTime": item.get("editTime") or item.get("createTime")
                    })
                return json.dumps({"type": "drafts_success", "count": len(simplified_drafts), "records": simplified_drafts}, ensure_ascii=False)
            else:
                return f'{{"error": "获取草稿列表接口失败，状态码: {draft_res.status_code}"}}'

        else:
            # 分页历史记录和列表数据
            # 拼接标准的通用分页查询入参
            payload = {
                "current": current,
                "pageSize": page_size,
                "sortField": "createTime",
                "sortOrder": "descend",
                "userId": user_id
            }

            api_path = ""
            if data_type == 'picture':
                api_path = "/picture/list/page/vo"
            elif data_type == 'post':
                api_path = "/post/my/list"
            elif data_type == 'likes':
                api_path = "/like/my/history"
            elif data_type == 'favorites':
                api_path = "/favorite-record/my/history"
            elif data_type == 'shares':
                api_path = "/share/my/history"
            elif data_type == 'comments':
                api_path = "/comments/my/history"
            else:
                return f'{{"error": "不支持的数据类型: {data_type}。可选的有 profile, picture, post, likes, favorites, shares, comments, drafts"}}'

            response = requests.post(f"{java_base_url}{api_path}", json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                res_json = response.json()
                if res_json.get("code") == 0:
                    page_data = res_json.get("data", {})
                    records = page_data.get("records", [])
                    total = page_data.get("total", 0)

                    # 智能字段简化，提取最有用的字段以压缩上下文大小
                    simplified_records = []
                    for item in records:
                        simplified_item = {
                            "id": item.get("id"),
                            "createTime": item.get("createTime")
                        }
                        if data_type == 'picture':
                            simplified_item.update({
                                "name": item.get("name"),
                                "url": item.get("url"),
                                "thumbnailUrl": item.get("thumbnailUrl"),
                                "category": item.get("category"),
                                "tags": item.get("tags")
                            })
                        elif data_type == 'post':
                            simplified_item.update({
                                "title": item.get("title"),
                                "content": item.get("content"),
                                "coverUrl": item.get("coverUrl"),
                                "tags": item.get("tags")
                            })
                        elif data_type in ['likes', 'favorites', 'shares']:
                            # 解析后端真实的嵌套 VO 对象（如 target、user）
                            target_obj = item.get("target") or {}
                            user_obj = item.get("user") or {}
                            
                            target_id = target_obj.get("id") or item.get("targetId") or item.get("pictureId") or item.get("postId")
                            target_type = item.get("targetType")
                            
                            title = target_obj.get("name") or target_obj.get("title") or item.get("title") or item.get("pictureName") or item.get("postTitle")
                            url = target_obj.get("url") or target_obj.get("thumbnailUrl") or item.get("url") or item.get("pictureUrl") or item.get("coverUrl")
                            
                            simplified_item.update({
                                "targetId": target_id,
                                "targetType": target_type,
                                "title": title,
                                "url": url,
                                "target": {
                                    "id": target_id,
                                    "name": title,
                                    "title": title,
                                    "thumbnailUrl": url,
                                    "url": url,
                                    "content": target_obj.get("content")
                                },
                                "user": {
                                    "userName": user_obj.get("userName") or "未知用户",
                                    "userAvatar": user_obj.get("userAvatar")
                                }
                            })
                        elif data_type == 'comments':
                            # 解析后端真实的嵌套 VO 对象（如 picture、post、commentUser）
                            picture_obj = item.get("picture") or {}
                            post_obj = item.get("post") or {}
                            comment_user = item.get("commentUser") or {}
                            
                            pic_id = picture_obj.get("id") or item.get("pictureId")
                            post_id = post_obj.get("id") or item.get("postId")
                            
                            simplified_item.update({
                                "content": item.get("content"),
                                "pictureId": pic_id,
                                "postId": post_id,
                                "replyCommentId": item.get("replyCommentId"),
                                "targetType": item.get("targetType"),
                                "picture": {
                                    "id": pic_id,
                                    "name": picture_obj.get("name") or item.get("pictureName") or picture_obj.get("title"),
                                    "thumbnailUrl": picture_obj.get("thumbnailUrl") or picture_obj.get("url"),
                                    "url": picture_obj.get("url")
                                },
                                "post": {
                                    "id": post_id,
                                    "title": post_obj.get("title") or post_obj.get("postTitle"),
                                    "content": post_obj.get("content")
                                },
                                "commentUser": {
                                    "userName": comment_user.get("userName") or "未知用户",
                                    "userAvatar": comment_user.get("userAvatar")
                                }
                            })
                        simplified_records.append(simplified_item)

                    return json.dumps({
                        "type": f"{data_type}_success",
                        "total": total,
                        "current": current,
                        "page_size": page_size,
                        "count": len(simplified_records),
                        "records": simplified_records
                    }, ensure_ascii=False)
                else:
                    return f'{{"error": "接口报错: {res_json.get("message")}"}}'
            else:
                return f'{{"error": "接口请求 HTTP 异常，状态码: {response.status_code}"}}'

    except Exception as e:
        knowledge_logger.error(f'[BIZ_TOOL_ERROR] 查询个人数据出现异常: {str(e)}')
        return f'{{"error": "调用查询个人数据接口失败: {str(e)}"}}'



@tool(description="""获取指定图片的详细信息（如图片分类、标签、简介、图片格式、图片主色调、作者信息等）。
当用户想查看某张图片的详细内容、了解详情或分析图片时必须使用此工具。
参数说明：
- picture_id (必须): 目标图片的 ID，可以通过全站搜索或获取个人数据列表得到。
""")
def get_picture_detail_data(picture_id: Union[int, str]) -> str:
    """获取单张图片的详细信息，并进行中文映射"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供有效凭证(sa-token)"}'
    
    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"
        
    headers = {"satoken": token, "Content-Type": "application/json"}
    try:
        knowledge_logger.info(f'[BIZ_TOOL] 正在获取图片详情 | ID: {picture_id}')
        res = requests.get(f"{java_base_url}/picture/get/vo?id={picture_id}", headers=headers, timeout=10)
        if res.status_code == 200:
            res_json = res.json()
            if res_json.get("code") != 0:
                return f'{{"error": "获取图片失败: {res_json.get("message")}"}}'
                
            data = res_json.get("data", {})
            if not data:
                return '{"error": "图片不存在或已被删除"}'
                
            mapped_data = {
                "图片ID": data.get("id"),
                "图片名称": data.get("name"),
                "图片简介": data.get("introduction"),
                "图片分类": data.get("category"),
                "图片标签": data.get("tags"),
                "图片大小": data.get("picSize"),
                "图片宽高": f"{data.get('picWidth')}x{data.get('picHeight')}",
                "图片比例": data.get("picScale"),
                "图片格式": data.get("picFormat"),
                "主色调": data.get("picColor"),
                "发布时间": data.get("createTime"),
                "作者信息": {
                    "作者ID": data.get("user", {}).get("id"),
                    "作者昵称": data.get("user", {}).get("userName"),
                    "作者账号": data.get("user", {}).get("userAccount")
                } if data.get("user") else "未知"
            }
            import json
            return json.dumps(mapped_data, ensure_ascii=False)
        return f'{{"error": "接口异常: {res.status_code}"}}'
    except Exception as e:
        return f'{{"error": "请求异常: {str(e)}"}}'

@tool(description="""获取指定帖子的详细信息（包含帖子标题、富文本内容、关联图片信息、作者信息等）。
当用户想看某个帖子的内容详情时使用。
参数说明：
- post_id (必须): 目标帖子的 ID。
""")
def get_post_detail_data(post_id: Union[int, str]) -> str:
    """获取单条帖子的详细信息，并进行中文映射"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供有效凭证(sa-token)"}'
    
    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"
        
    headers = {"satoken": token, "Content-Type": "application/json"}
    try:
        knowledge_logger.info(f'[BIZ_TOOL] 正在获取帖子详情 | ID: {post_id}')
        res = requests.get(f"{java_base_url}/post/get/{post_id}", headers=headers, timeout=10)
        if res.status_code == 200:
            res_json = res.json()
            if res_json.get("code") != 0:
                return f'{{"error": "获取帖子失败: {res_json.get("message")}"}}'
                
            data = res_json.get("data", {})
            if not data:
                return '{"error": "帖子不存在或已被删除"}'
                
            mapped_data = {
                "帖子ID": data.get("id"),
                "帖子标题": data.get("title"),
                "帖子正文": data.get("content"),
                "封面图片": data.get("coverUrl"),
                "帖子分类": data.get("category"),
                "帖子标签": data.get("tags"),
                "点赞数": data.get("likeCount", 0),
                "收藏数": data.get("favoriteCount", 0),
                "分享数": data.get("shareCount", 0),
                "评论数": data.get("commentCount", 0),
                "发布时间": data.get("createTime"),
                "作者信息": {
                    "作者ID": data.get("user", {}).get("id"),
                    "作者昵称": data.get("user", {}).get("userName"),
                    "作者账号": data.get("user", {}).get("userAccount")
                } if data.get("user") else "未知"
            }
            import json
            return json.dumps(mapped_data, ensure_ascii=False)
        return f'{{"error": "接口异常: {res.status_code}"}}'
    except Exception as e:
        return f'{{"error": "请求异常: {str(e)}"}}'

@tool(description="""获取指定用户的详细公开资料（包含用户名、头像、性别、生日、地区、个人签名等）。
当用户想要查看自己或他人的详细资料详情时使用此工具。
参数说明：
- user_id (必须): 目标用户的 ID。
""")
def get_user_detail_data(user_id: Union[int, str]) -> str:
    """获取用户的详细资料，并进行中文映射"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供有效凭证(sa-token)"}'
    
    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"
        
    headers = {"satoken": token, "Content-Type": "application/json"}
    try:
        knowledge_logger.info(f'[BIZ_TOOL] 正在获取用户详情 | ID: {user_id}')
        res = requests.get(f"{java_base_url}/user/get/vo?id={user_id}", headers=headers, timeout=10)
        if res.status_code == 200:
            res_json = res.json()
            if res_json.get("code") != 0:
                return f'{{"error": "获取用户详情失败: {res_json.get("message")}"}}'
                
            data = res_json.get("data", {})
            if not data:
                return '{"error": "用户不存在"}'
                
            gender_map = {0: "男", 1: "女"}
            mapped_data = {
                "用户ID": data.get("id"),
                "用户账号": data.get("userAccount"),
                "用户昵称": data.get("userName"),
                "用户头像": data.get("userAvatar"),
                "用户简介": data.get("userProfile"),
                "用户性别": gender_map.get(data.get("gender"), "未知"),
                "用户生日": data.get("birthday"),
                "所属地区": data.get("region"),
                "个人签名": data.get("personalSign"),
                "用户标签": data.get("userTags"),
                "用户角色": data.get("userRole"),
                "注册时间": data.get("createTime")
            }
            import json
            return json.dumps(mapped_data, ensure_ascii=False)
        return f'{{"error": "接口异常: {res.status_code}"}}'
    except Exception as e:
        return f'{{"error": "请求异常: {str(e)}"}}'

@tool(description="""获取指定用户的关注列表或粉丝列表（分页数据）。
当用户说"查看我的关注"、"看看是谁关注了我"或查看指定人的粉丝/关注时使用。
参数说明：
- user_id (必须): 目标用户的 ID。如果是查自己，需要传入自己的 ID。
- list_type (必须): 查询类型，值为 "follow"（关注列表） 或 "fans"（粉丝列表）。
- current (可选): 页码，默认为 1。
- page_size (可选): 每页大小，默认为 10。
""")
def get_follow_or_fan_list_data(user_id: Union[int, str], list_type: str, current: int = 1, page_size: int = 10) -> str:
    """获取指定用户的关注或粉丝列表，并进行中文映射"""
    token = get_sa_token()
    if not token:
        return '{"error": "未提供有效凭证(sa-token)"}'
    
    try:
        from model.factory import load_config
        config = load_config()
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
    except Exception:
        java_base_url = "http://127.0.0.1:8123/api"
        
    headers = {"satoken": token, "Content-Type": "application/json"}
    
    search_type = 0 if list_type == 'follow' else 1
    
    payload = {
        "current": current,
        "pageSize": page_size,
        "searchType": search_type
    }
    
    if search_type == 0:
        payload["followerId"] = user_id
    else:
        payload["followingId"] = user_id
        
    try:
        knowledge_logger.info(f'[BIZ_TOOL] 正在获取关注/粉丝列表 | ID: {user_id} | 类型: {list_type}')
        res = requests.post(f"{java_base_url}/userfollows/getfolloworfanlist", json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            res_json = res.json()
            if res_json.get("code") != 0:
                return f'{{"error": "获取列表失败: {res_json.get("message")}"}}'
                
            data = res_json.get("data", {})
            if not data:
                return '{"error": "列表为空或无权限查看"}'
                
            records = data.get("records", [])
            mapped_records = []
            for item in records:
                mapped_records.append({
                    "用户ID": item.get("id"),
                    "用户账号": item.get("userAccount"),
                    "用户昵称": item.get("userName"),
                    "用户头像": item.get("userAvatar"),
                    "用户简介": item.get("userProfile")
                })
                
            result = {
                "总记录数": data.get("total", 0),
                "当前页码": data.get("current", 1),
                "每页大小": data.get("size", 10),
                "列表数据": mapped_records
            }
            import json
            return json.dumps(result, ensure_ascii=False)
        return f'{{"error": "接口异常: {res.status_code}"}}'
    except Exception as e:
        return f'{{"error": "请求异常: {str(e)}"}}'
