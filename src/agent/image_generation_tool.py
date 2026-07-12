import os
import requests
from langchain_core.tools import tool
from model.factory import load_config
from utils.log_utils import app_logger

@tool
def generate_image(prompt: str) -> str:
    """
    当用户要求生成图片、画一张图、或创作一幅画时，必须调用此工具。
    Z-Image-Turbo 是一款世界领先的高效图像生成模型，支持中英双语，能生成媲美商业模型的照片级真实感图像。
    
    参数:
        prompt: 描述要生成的图像的详细提示词，越详细越好（支持中文或英文）。如果用户提示较短，你可以自行扩写以增加细节。
    返回:
        成功时返回生成图片的 Markdown 格式字符串，失败时返回错误信息。
    """
    try:
        config = load_config()
        api_key = config.get('qwen_api_key')
        if not api_key:
            return "错误：未配置 DashScope API Key，请在 .env 文件中设置 QWEN_API_KEY 环境变量"
            
        # --- 1. 额度预扣减逻辑 ---
        from .context import get_sa_token
        token = get_sa_token()
        if not token:
            return "错误：未提供身份凭证(sa-token)，无法执行生图额度校验。"
            
        java_base_url = config.get("java_backend_url", "http://127.0.0.1:8123/api")
        quota_url = f"{java_base_url}/rag/qa/message/image_gen/quota/deduct"
        quota_headers = {
            "satoken": token,
            "Content-Type": "application/json"
        }
        try:
            app_logger.info(f"[IMAGE_GEN] 正在预扣减图片生成额度...")
            quota_res = requests.post(quota_url, headers=quota_headers, timeout=5)
            if quota_res.status_code == 200:
                quota_result = quota_res.json()
                if quota_result.get("code") != 0:
                    return f"❌ 额度校验失败：{quota_result.get('message')}（普通用户每周5次，Pro用户15次，Plus用户30次）"
            else:
                return f"❌ 无法连接到额度校验服务（HTTP {quota_res.status_code}）"
        except Exception as e:
            app_logger.error(f"[IMAGE_GEN] 额度扣减请求异常: {str(e)}")
            return f"❌ 额度校验异常：{str(e)}"
            
        # --- 2. 扣减成功，正式生成图片 ---
        url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        # 默认生成比例 1024*1024，也可以尝试 1120*1440
        data = {
            'model': 'z-image-turbo',
            'input': {
                'messages': [
                    {
                        'role': 'user',
                        'content': [{'text': prompt}]
                    }
                ]
            },
            'parameters': {
                'prompt_extend': False, # 关闭模型端 prompt 自动扩写，使用标准模式（0.1元/张）
                'size': '1024*1024'
            }
        }
        
        app_logger.info(f"[IMAGE_GEN] 开始调用 Z-Image-Turbo 生成图片 | prompt: {prompt[:50]}...")
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            res_json = response.json()
            try:
                # 解析返回结果
                content_list = res_json['output']['choices'][0]['message']['content']
                img_url = None
                for item in content_list:
                    if 'image' in item:
                        img_url = item['image']
                        break
                        
                if img_url:
                    app_logger.info(f"[IMAGE_GEN] 生成成功，准备上传至图库: {img_url}")
                    
                    try:
                        # 3. 将生成的临时OSS链接下载并上传到我们的图库(COS)
                        import uuid
                        import tempfile
                        from .image_upload_tool import upload_local_image
                        
                        # 下载临时图片
                        temp_dir = tempfile.gettempdir()
                        local_path = os.path.join(temp_dir, f"ai_gen_{uuid.uuid4().hex}.jpg")
                        
                        img_res = requests.get(img_url, timeout=30)
                        if img_res.status_code == 200:
                            with open(local_path, "wb") as f:
                                f.write(img_res.content)
                                
                            app_logger.info(f"[IMAGE_GEN] 临时图片下载完成，正在上传至腾讯云 COS...")
                            
                            # 调用图片上传工具，space_id=None 代表上传到公共空间，name为提示词的前30个字符
                            upload_result_str = upload_local_image.invoke({
                                "image_path": local_path,
                                "space_id": None,
                                "name": prompt[:30],
                                "introduction": f"AI生成图片: {prompt}"
                            })
                            
                            import json
                            try:
                                upload_result = json.loads(upload_result_str)
                                if "url" in upload_result:
                                    final_url = upload_result["url"]
                                    app_logger.info(f"[IMAGE_GEN] 上传腾讯云成功: {final_url}")
                                    return f"图片生成并永久保存成功：\n\n![{prompt}]({final_url})\n\n*(提示词: {prompt})*"
                                else:
                                    app_logger.warning(f"[IMAGE_GEN] 上传失败: {upload_result_str}")
                            except Exception as e:
                                app_logger.error(f"[IMAGE_GEN] 解析上传结果失败: {e}")
                            
                            # 清理临时文件
                            try:
                                os.remove(local_path)
                            except:
                                pass
                                
                        else:
                            app_logger.warning(f"[IMAGE_GEN] 无法下载生成的临时图片，状态码: {img_res.status_code}")
                            
                    except Exception as e:
                        app_logger.error(f"[IMAGE_GEN] 下载或上传过程中出错: {e}")
                        
                    # 如果上传失败，降级返回原有的临时链接
                    app_logger.warning("[IMAGE_GEN] 上传腾讯云失败，返回临时 OSS 链接")
                    return f"图片生成成功（临时链接，请尽快保存）：\n\n![{prompt}]({img_url})\n\n*(提示词: {prompt})*"
                else:
                    return f"生成成功，但未解析到图片地址。返回原文：{response.text}"
            except Exception as e:
                return f"解析生成结果异常：{str(e)}，返回原文：{response.text}"
        else:
            app_logger.error(f"[IMAGE_GEN] 生成失败 | 状态码: {response.status_code} | 响应: {response.text}")
            return f"图片生成失败，请稍后重试。（错误码：{response.status_code}）"
            
    except Exception as e:
        app_logger.error(f"[IMAGE_GEN] 工具执行异常: {str(e)}")
        return f"图片生成异常：{str(e)}"
