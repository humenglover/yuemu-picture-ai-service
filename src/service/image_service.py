import os
import shutil
import cv2
import numpy as np
from utils.log_utils import app_logger

class ImageService:
    def __init__(self, detect_service):
        """
        初始化图片处理服务
        :param detect_service: NanoDetService 实例（原用于检测辅助）
        """
        self.detect_service = detect_service
        
        # 参考目标检测方式，直接用 OpenCV DNN 加载本地 u2netp.onnx 模型，彻底摆脱 rembg 和网络下载
        try:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            model_path = os.path.join(model_dir, 'u2netp.onnx')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"未找到 u2netp.onnx 模型文件：{model_path}")
                
            app_logger.info(f"正在加载 u2netp ONNX 模型: {model_path}")
            
            # 解决 Windows 下含中文路径的问题
            with open(model_path, "rb") as f:
                model_buffer = bytearray(f.read())
            self.u2net = cv2.dnn.readNetFromONNX(model_buffer)
            app_logger.info("u2netp ONNX 模型加载成功 (OpenCV DNN)")
            
        except Exception as e:
            app_logger.error(f"初始化 u2netp 模型失败: {str(e)}")
            raise RuntimeError(f"无法加载抠图模型: {str(e)}")

    def remove_background(self, input_image_bytes: bytes) -> bytes:
        """
        利用纯 OpenCV DNN 加载 u2netp 模型进行高精度抠图
        """
        try:
            app_logger.info("开始使用 UNet (u2netp) 模型进行抠图 (OpenCV DNN)...")
            
            nparr = np.frombuffer(input_image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("无法解析图像数据")
                
            orig_h, orig_w = img.shape[:2]
            
            # 1. 预处理 (与官方 U2Net 保持一致)
            img_resized = cv2.resize(img, (320, 320))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_f = img_rgb.astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_f = (img_f - mean) / std
            
            # [H, W, C] -> [N, C, H, W]
            blob = img_f.transpose(2, 0, 1)[np.newaxis, ...]
            
            # 2. 推理
            self.u2net.setInput(blob)
            outs = self.u2net.forward(self.u2net.getUnconnectedOutLayersNames())
            
            # U2Net 取第一个输出 d0
            pred = outs[0][0, 0, :, :]
            
            # 3. 后处理 (Min-Max 归一化)
            ma = np.max(pred)
            mi = np.min(pred)
            if ma > mi:
                pred = (pred - mi) / (ma - mi)
                
            mask = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask = (mask * 255).astype(np.uint8)
            
            # 4. 生成带 Alpha 通道的 PNG
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask
            
            _, buf = cv2.imencode('.png', rgba)
            app_logger.info("抠图成功")
            return buf.tobytes()
            
        except Exception as e:
            app_logger.error(f"Unet 抠图失败: {e}")
            raise e

    def change_background(self, input_image_bytes: bytes, background_color: str = None, background_image_bytes: bytes = None) -> bytes:
        raise NotImplementedError("背景替换服务已被手动下线（请使用前端纯色替换或独立重构）。")

    def blur_faces(self, input_image_bytes: bytes) -> bytes:
        """
        利用 OpenCV 自带的 Haar 级联分类器进行人脸打码（零外部模型依赖）
        """
        # 将 bytes 转为 numpy 数组 (BGR)
        nparr = np.frombuffer(input_image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解析图像数据")
        
        # 转换为灰度图以提升检测速度和准确率
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 加载 OpenCV 内置的人脸级联分类器
        cascade_src = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        cascade_path = cascade_src
        
        # 解决 OpenCV C++ 底层无法读取带中文的绝对路径的 Bug
        if os.name == 'nt' and not cascade_path.isascii():
            safe_dir = r"C:\Users\Public"
            if os.path.exists(safe_dir):
                cascade_path = os.path.join(safe_dir, "haarcascade_frontalface_default.xml")
                if not os.path.exists(cascade_path):
                    shutil.copy2(cascade_src, cascade_path)
                    
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 多尺度人脸检测
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        app_logger.info(f"OpenCV Haar 人脸检测到 {len(faces)} 张人脸")
        
        # 遍历每一张人脸，应用马赛克/高斯模糊
        for (x, y, w, h) in faces:
            # 提取人脸 ROI
            face_roi = img[y:y+h, x:x+w]
            # 使用强力的高斯模糊进行打码
            # 核大小根据人脸尺寸自适应，保证足够的模糊度
            ksize = (w // 3 | 1, h // 3 | 1) # 必须为奇数
            blurred_face = cv2.GaussianBlur(face_roi, ksize, 0)
            
            # 创建一个椭圆形 mask 模拟人脸轮廓
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            center = (w // 2, h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
            
            # 对 mask 边缘进行高斯羽化，让打码边缘平滑过渡
            mask_blur_ksize = (w // 5 | 1, h // 5 | 1)
            mask_blur = cv2.GaussianBlur(mask, mask_blur_ksize, 0)
            
            # 将 mask 归一化为 Alpha 通道权重 (0.0 ~ 1.0)
            alpha = mask_blur.astype(float) / 255.0
            
            # 按照 Alpha 权重融合模糊图像与原图像
            blended = (alpha * blurred_face + (1 - alpha) * face_roi).astype(np.uint8)
            
            # 将融合后的人脸替换回原图
            img[y:y+h, x:x+w] = blended
        # 转回 bytes 返回
        _, buf = cv2.imencode('.jpg', img)
        return buf.tobytes()

    def enhance_image(self, input_image_bytes: bytes) -> bytes:
        """
        零依赖增强清晰度：使用 Unsharp Masking (USM) + 局部对比度增强
        """
        nparr = np.frombuffer(input_image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解析图像数据")
            
        # 1. 细节增强 (HDR-like detail enhancement)
        enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        
        # 2. 轻微的 Unsharp Mask 锐化
        gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.2, gaussian_blur, -0.2, 0)
        
        _, buf = cv2.imencode('.jpg', sharpened)
        return buf.tobytes()
