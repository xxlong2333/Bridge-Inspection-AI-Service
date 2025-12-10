"""
YOLOv8检测器封装
 支持图像和视频流的实时检测，带车辆跟踪功能
"""
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Generator
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from loguru import logger


class VehicleTracker:
    """简单的车辆跟踪器 - 基于YOLO内置的ByteTrack"""
    
    def __init__(self):
        self.tracked_ids = set()  # 已跟踪过的车辆ID
        self.vehicle_counts = {}  # 各类型车辆计数
        self.total_count = 0
        
    def reset(self):
        """重置跟踪器"""
        self.tracked_ids.clear()
        self.vehicle_counts.clear()
        self.total_count = 0
        
    def update(self, detections: List[Dict[str, Any]]) -> Tuple[int, Dict[str, int]]:
        """
        更新跟踪状态，返回新增车辆数
        
        Args:
            detections: 带track_id的检测结果列表
            
        Returns:
            (新增车辆数, 新增各类型车辆数)
        """
        new_vehicles = 0
        new_by_type = {}
        
        for det in detections:
            track_id = det.get('track_id')
            if track_id is not None and track_id not in self.tracked_ids:
                self.tracked_ids.add(track_id)
                self.total_count += 1
                new_vehicles += 1
                
                class_name = det.get('class_name', 'unknown')
                self.vehicle_counts[class_name] = self.vehicle_counts.get(class_name, 0) + 1
                new_by_type[class_name] = new_by_type.get(class_name, 0) + 1
                
        return new_vehicles, new_by_type
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'unique_vehicle_count': self.total_count,
            'vehicle_count_by_type': self.vehicle_counts.copy(),
            'tracked_ids_count': len(self.tracked_ids)
        }


class YOLODetector:
    """YOLOv8检测器封装类"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        use_half: bool = True,
        class_names: Optional[Dict[int, str]] = None,
        visualization_colors: Optional[Dict[str, List[int]]] = None
    ):
        """
        初始化YOLO检测器

        Args:
            model_path: 模型权重文件路径
            device: 设备 (cuda, cpu, mps)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU阈值
            max_det: 最大检测数量
            use_half: 是否使用FP16半精度
            class_names: 类别名称字典
            visualization_colors: 可视化颜色配置字典 (BGR格式)
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.use_half = use_half
        self.class_names = class_names or {}
        # 存储颜色配置，转换为元组格式
        self.visualization_colors = {}
        if visualization_colors:
            for name, color in visualization_colors.items():
                self.visualization_colors[name] = tuple(color) if isinstance(color, list) else color

        # 加载模型
        self._load_model()

        logger.info(f"YOLODetector initialized on {self.device}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Classes: {self.class_names}")

    def _load_model(self):
        """加载YOLO模型"""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            self.model = YOLO(self.model_path)

            # 设置设备
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"

            # 将模型移动到指定设备
            self.model.to(self.device)

            # 使用半精度（仅GPU支持）
            if self.use_half and self.device == "cuda":
                self.model.model.half()
                logger.info("FP16 half precision enabled")

            # 预热模型
            logger.info("Warming up model...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(
                dummy_img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            logger.success("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect_image(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        检测单张图像

        Args:
            image: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值（可选，覆盖默认值）
            iou_threshold: IOU阈值（可选，覆盖默认值）

        Returns:
            检测结果字典
        """
        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold

        start_time = time.time()

        try:
            # 执行推理
            results = self.model.predict(
                image,
                conf=conf,
                iou=iou,
                max_det=self.max_det,
                device=self.device,
                verbose=False
            )

            inference_time = time.time() - start_time

            # 解析结果
            detections = self._parse_results(results[0])
            # print(detections)

            return {
                "success": True,
                "detections": detections,
                "inference_time": round(inference_time, 4),
                "detection_count": len(detections),
                "image_shape": image.shape[:2]
            }

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "inference_time": 0,
                "detection_count": 0
            }

    def detect_video_frame(
        self,
        frame: np.ndarray,
        draw_results: bool = True,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        检测视频帧

        Args:
            frame: 视频帧 (BGR格式)
            draw_results: 是否在帧上绘制检测结果
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            (标注后的帧, 检测结果字典)
        """
        # 执行检测
        result = self.detect_image(frame, conf_threshold, iou_threshold)

        # 绘制结果
        if draw_results and result["success"]:
            frame = self.draw_detections(frame, result["detections"])

        return frame, result

    def _parse_results(self, result) -> List[Dict[str, Any]]:
        """
        解析YOLO检测结果

        Args:
            result: YOLO结果对象

        Returns:
            检测列表
        """
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            class_id = int(class_ids[i])
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            confidence = float(confidences[i])
            bbox = boxes[i].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(x, 2) for x in bbox],  # [x1, y1, x2, y2]
                "bbox_center": [
                    round((bbox[0] + bbox[2]) / 2, 2),
                    round((bbox[1] + bbox[3]) / 2, 2)
                ],
                "bbox_area": round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 2)
            })

        return detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        box_thickness: int = 2,
        font_scale: float = 0.6,
        show_conf: bool = True,
        show_class: bool = True,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 输入图像
            detections: 检测结果列表
            box_thickness: 边界框线条粗细
            font_scale: 字体大小
            show_conf: 是否显示置信度
            show_class: 是否显示类别名称
            colors: 类别颜色字典 (BGR格式)

        Returns:
            标注后的图像
        """
        img = image.copy()

        # 默认颜色
        default_colors = {
            "corrosion_stain": (0, 255, 255),       # 青色 (BGR)
            "crack": (0, 0, 255),                    # 红色
            "efflorescence": (0, 255, 0),            # 绿色
            "exposed_reinforcement": (255, 0, 255),  # 紫色
            "spalling": (255, 255, 0),               # 黄色 
        }
        # 优先使用实例化时传入的颜色配置
        if colors:
            colors = colors
        elif self.visualization_colors:
            colors = self.visualization_colors
        else:
            colors = default_colors

        for det in detections:
            class_name = det["class_name"]
            confidence = det["confidence"]
            bbox = det["bbox"]

            # 获取颜色
            color = colors.get(class_name, (255, 255, 255))

            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

            # 构建标签文本
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_conf:
                label_parts.append(f"{confidence:.2f}")

            label = " ".join(label_parts)

            # 绘制标签背景
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            cv2.rectangle(
                img,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )

            # 绘制标签文本
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2
            )

        return img

    def track_video_frame(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        persist: bool = True
    ) -> Dict[str, Any]:
        """
        使用ByteTrack跟踪视频帧中的车辆
        
        Args:
            frame: 视频帧 (BGR格式)
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            persist: 是否持久化跟踪状态（跨帧跟踪）
            
        Returns:
            带跟踪ID的检测结果字典
        """
        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold
        
        start_time = time.time()
        
        try:
            # 使用track方法进行目标跟踪
            results = self.model.track(
                frame,
                conf=conf,
                iou=iou,
                max_det=self.max_det,
                device=self.device,
                persist=persist,
                tracker="bytetrack.yaml",
                verbose=False
            )
            
            inference_time = time.time() - start_time
            
            # 解析带跟踪ID的结果
            detections = self._parse_track_results(results[0])
            
            return {
                "success": True,
                "detections": detections,
                "inference_time": round(inference_time, 4),
                "detection_count": len(detections),
                "image_shape": frame.shape[:2]
            }
            
        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "inference_time": 0,
                "detection_count": 0
            }
    
    def _parse_track_results(self, result) -> List[Dict[str, Any]]:
        """
        解析带跟踪ID的YOLO结果
        
        Args:
            result: YOLO track结果对象
            
        Returns:
            带track_id的检测列表
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # 获取跟踪ID（如果存在）
        track_ids = None
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        
        for i in range(len(boxes)):
            class_id = int(class_ids[i])
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            confidence = float(confidences[i])
            bbox = boxes[i].tolist()
            
            det = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(x, 2) for x in bbox],
                "bbox_center": [
                    round((bbox[0] + bbox[2]) / 2, 2),
                    round((bbox[1] + bbox[3]) / 2, 2)
                ],
                "bbox_area": round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 2)
            }
            
            # 添加跟踪ID
            if track_ids is not None:
                det["track_id"] = int(track_ids[i])
            
            detections.append(det)
        
        return detections
    
    def reset_tracker(self):
        """重置跟踪器状态"""
        # 重新加载模型会清除跟踪状态
        # 或者可以通过model.predictor.trackers[0].reset()来重置
        try:
            if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                if hasattr(self.model.predictor, 'trackers'):
                    for tracker in self.model.predictor.trackers:
                        if hasattr(tracker, 'reset'):
                            tracker.reset()
        except Exception as e:
            logger.warning(f"Failed to reset tracker: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_det": self.max_det,
            "use_half": self.use_half,
            "class_names": self.class_names,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
