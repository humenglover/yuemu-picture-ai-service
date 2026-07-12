import cv2
import numpy as np
import base64
from typing import Dict, Any
import os
from utils.log_utils import app_logger


class NanoDetService:
    """
    NanoDet-Plus 目标检测服务（纯 ONNX 推理）
    支持模型：nanodet-plus-m_320.onnx (~3.7MB) / nanodet-plus-m_416.onnx
    输出格式：[1, N, 4*(reg_max+1) + num_classes] 单 tensor，其中 N = 2125（320x320，strides=[8,16,32,64]）
    """

    STRIDES = [8, 16, 32, 64]
    REG_MAX = 7          # GFL 分布离散化区间数，官方默认为 7

    # COCO 80 类（与原 YOLOv8 保持一致）
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, model_path: str):
        """
        初始化 NanoDet-Plus 服务
        :param model_path: .onnx 模型文件的绝对路径
        """
        self.model_path = model_path
        app_logger.info(f"正在加载 NanoDet-Plus ONNX 模型: {model_path}")

        try:
            # OpenCV DNN 在 Windows 下直接读取含中文路径的文件会失败，所以先读入内存
            with open(model_path, "rb") as f:
                model_buffer = bytearray(f.read())
            self.net = cv2.dnn.readNetFromONNX(model_buffer)

            # OpenCV DNN 获取 ONNX 动态 input shape 不太方便，NanoDet 常用尺寸为 320 或 416
            if "416" in model_path:
                self.input_h, self.input_w = 416, 416
            else:
                self.input_h, self.input_w = 320, 320

            # 预计算所有 stride 的 anchor 中心点
            self.centers = self._generate_centers()

            app_logger.info(
                f"NanoDet-Plus ONNX 模型加载成功 (OpenCV DNN) "
                f"(输入尺寸: {self.input_h}x{self.input_w}, anchor数: {len(self.centers)})"
            )
        except Exception as e:
            app_logger.error(f"NanoDet-Plus ONNX 模型加载失败 (OpenCV DNN): {str(e)}")
            raise e

    # ------------------------------------------------------------------ #
    #  预处理                                                               #
    # ------------------------------------------------------------------ #

    def _generate_centers(self) -> np.ndarray:
        """预计算各 stride 下所有 anchor 中心点坐标 [N, 3] (cx, cy, stride)"""
        centers = []
        for stride in self.STRIDES:
            fh = self.input_h // stride
            fw = self.input_w // stride
            for i in range(fh):
                for j in range(fw):
                    centers.append([(j + 0.5) * stride, (i + 0.5) * stride, stride])
        return np.array(centers, dtype=np.float32)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        NanoDet-Plus 标准预处理：
          1. resize 到模型输入尺寸（直接 resize，不做 letterbox）
          2. 用 ImageNet BGR 均值/标准差归一化（[0,255] 值域）
          3. HWC -> NCHW
        """
        img_resized = cv2.resize(img, (self.input_w, self.input_h))
        img_f = img_resized.astype(np.float32)

        # NanoDet-Plus 使用 BGR 顺序的 ImageNet 统计量（×255 尺度）
        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)  # BGR
        std  = np.array([57.375,  57.12,  58.395], dtype=np.float32)  # BGR
        img_f = (img_f - mean) / std

        return img_f.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # [1,3,H,W]

    # ------------------------------------------------------------------ #
    #  GFL 解码                                                            #
    # ------------------------------------------------------------------ #

    def _decode_gfl(self, pred_dist: np.ndarray) -> np.ndarray:
        """
        将 GFL (Generalized Focal Loss) 分布解码为 ltrb 格式的边界框
        :param pred_dist: [N, 4*(reg_max+1)]
        :return:          [N, 4]  ltrb，单位为 stride（像素坐标需再乘以 stride）
        """
        n = pred_dist.shape[0]
        k = self.REG_MAX + 1
        # reshape 为 [N, 4, k]，对每个方向独立做 softmax 求期望
        dist = pred_dist.reshape(n, 4, k)
        dist = np.exp(dist - dist.max(axis=-1, keepdims=True))
        dist /= dist.sum(axis=-1, keepdims=True)
        weights = np.arange(k, dtype=np.float32)
        return (dist * weights).sum(axis=-1)   # [N, 4]

    # ------------------------------------------------------------------ #
    #  推理主入口                                                           #
    # ------------------------------------------------------------------ #

    def detect(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        对图像字节流进行目标检测
        :param image_bytes: 原始图像字节（jpg/png 等）
        :return: {"detections": [...], "annotatedImageBase64": "..."}
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解析图像数据")

        orig_h, orig_w = img.shape[:2]
        input_tensor = self.preprocess(img)

        # ---- OpenCV DNN 推理 ----
        self.net.setInput(input_tensor)
        out_names = self.net.getUnconnectedOutLayersNames()
        raw_outputs = self.net.forward(out_names)

        # 兼容两种常见导出格式：
        #   (A) 单 output: [1, N, 4*(reg_max+1)+num_classes]
        #   (B) 多 output: 6 个独立 tensor（官方 export.py 默认格式）
        if len(raw_outputs) == 1:
            pred = raw_outputs[0][0]  # [N, 112]
        else:
            pred = self._merge_multi_outputs(raw_outputs)

        # ---- 分离类别分数 与 bbox 分布 ----
        # NanoDet-Plus 输出格式： [cls(80)在前, bbox_dist(32)在后]
        num_cls = len(self.CLASSES)  # 80
        k = self.REG_MAX + 1         # 8
        cls_raw   = pred[:, :num_cls]       # [N, 80]
        pred_dist = pred[:, num_cls:]       # [N, 32]

        # NanoDet-Plus ONNX 内部已包含 sigmoid，直接使用原始输出
        cls_score = cls_raw  # 已在 [0, 1] 范围，无需再次激活

        max_scores = cls_score.max(axis=1)   # [N]
        class_ids  = cls_score.argmax(axis=1)  # [N]

        # ---- 置信度阈值过滤 ----
        conf_thresh = 0.40
        nms_thresh  = 0.40
        mask = max_scores >= conf_thresh

        if not mask.any():
            _, buf = cv2.imencode('.jpg', img)
            return {"detections": [], "annotatedImageBase64": base64.b64encode(buf).decode('utf-8')}

        f_dist     = pred_dist[mask]
        f_scores   = max_scores[mask]
        f_cls      = class_ids[mask]
        f_centers  = self.centers[mask]   # [M, 3]

        # ---- GFL 解码 -> ltrb (stride 单位) -> 像素 xyxy ----
        ltrb    = self._decode_gfl(f_dist)            # [M, 4]
        strides = f_centers[:, 2:3]                   # [M, 1]
        ltrb_px = ltrb * strides                      # [M, 4]  像素 ltrb

        cx, cy = f_centers[:, 0], f_centers[:, 1]
        x1 = cx - ltrb_px[:, 0]
        y1 = cy - ltrb_px[:, 1]
        x2 = cx + ltrb_px[:, 2]
        y2 = cy + ltrb_px[:, 3]

        # 缩放回原图尺寸
        sx, sy = orig_w / self.input_w, orig_h / self.input_h
        x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy

        # 转换为 [x, y, w, h] 整型，供 NMSBoxes 使用
        boxes_xywh = []
        for i in range(len(x1)):
            bx, by = int(x1[i]), int(y1[i])
            bw, bh = int(x2[i] - x1[i]), int(y2[i] - y1[i])
            boxes_xywh.append([bx, by, bw, bh])

        # ---- NMS (Class-agnostic，全局合并) ----
        indices = cv2.dnn.NMSBoxes(boxes_xywh, f_scores.tolist(), conf_thresh, nms_thresh)

        detections = []
        if len(indices) > 0:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            for i in indices:
                box   = boxes_xywh[i]
                label = (self.CLASSES[f_cls[i]]
                         if f_cls[i] < len(self.CLASSES)
                         else f"class_{f_cls[i]}")
                detections.append({
                    "label":      label,
                    "confidence": float(f_scores[i]),
                    "x":          box[0],
                    "y":          box[1],
                    "width":      box[2],
                    "height":     box[3],
                })
                # 绘制检测框
                bx, by, bw, bh = box
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(
                    img, f"{label} {f_scores[i]:.2f}",
                    (max(0, bx), max(10, by - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        _, buf = cv2.imencode('.jpg', img)
        return {
            "detections":          detections,
            "annotatedImageBase64": base64.b64encode(buf).decode('utf-8'),
        }

    def detect_from_url(self, url: str) -> Dict[str, Any]:
        """从 URL 下载图像后进行检测（兼容旧接口）"""
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return self.detect(resp.content)

    # ------------------------------------------------------------------ #
    #  多输出格式兼容（官方 export.py 导出时的 6-tensor 格式）              #
    # ------------------------------------------------------------------ #

    def _merge_multi_outputs(self, raw_outputs) -> np.ndarray:
        """
        将官方 6-output 格式（cls×3 + bbox×3）拼合为单个 [N, 112] tensor
        每个 stride 输出 shape: cls=[1,80,H,W], bbox=[1,32,H,W]
        """
        parts = []
        # 假设顺序：stride8_cls, stride8_bbox, stride16_cls, stride16_bbox, ...
        # 若模型输出顺序不同，需调整索引
        num_heads = len(self.STRIDES)
        for idx in range(num_heads):
            cls_t  = raw_outputs[idx * 2]      # [1, 80, H, W]
            bbox_t = raw_outputs[idx * 2 + 1]  # [1, 32, H, W]
            # Flatten 空间维度: [1, C, H, W] -> [H*W, C]
            cls_flat  = cls_t[0].reshape(cls_t.shape[1], -1).T
            bbox_flat = bbox_t[0].reshape(bbox_t.shape[1], -1).T
            parts.append(np.concatenate([bbox_flat, cls_flat], axis=1))  # [H*W, 112]
        return np.concatenate(parts, axis=0)  # [N, 112]
