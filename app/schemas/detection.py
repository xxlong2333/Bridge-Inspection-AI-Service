"""
API请求和响应模型
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """边界框"""
    x1: float = Field(..., description="左上角X坐标")
    y1: float = Field(..., description="左上角Y坐标")
    x2: float = Field(..., description="右下角X坐标")
    y2: float = Field(..., description="右下角Y坐标")


class Detection(BaseModel):
    """单个检测结果"""
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., description="置信度", ge=0, le=1)
    bbox: List[float] = Field(..., description="边界框 [x1, y1, x2, y2]")
    bbox_center: List[float] = Field(..., description="边界框中心点 [x, y]")
    bbox_area: float = Field(..., description="边界框面积")


class DetectionRequest(BaseModel):
    """图像检测请求"""
    conf_threshold: Optional[float] = Field(0.25, description="置信度阈值", ge=0, le=1)
    iou_threshold: Optional[float] = Field(0.45, description="NMS IOU阈值", ge=0, le=1)
    return_image: bool = Field(False, description="是否返回标注后的图像")
    image_format: str = Field("JPEG", description="返回图像格式 (JPEG, PNG)")


class DetectionResponse(BaseModel):
    """图像检测响应"""
    success: bool = Field(..., description="是否成功")
    detections: List[Detection] = Field(default=[], description="检测结果列表")
    detection_count: int = Field(..., description="检测到的目标数量")
    inference_time: float = Field(..., description="推理时间（秒）")
    image_shape: Optional[List[int]] = Field(None, description="图像尺寸 [height, width]")
    result_image: Optional[str] = Field(None, description="标注后的图像（Base64）")
    error: Optional[str] = Field(None, description="错误信息")


class VideoStreamRequest(BaseModel):
    """视频流检测请求"""
    stream_url: str = Field(..., description="视频流URL")
    conf_threshold: Optional[float] = Field(0.25, description="置信度阈值", ge=0, le=1)
    iou_threshold: Optional[float] = Field(0.45, description="NMS IOU阈值", ge=0, le=1)
    frame_skip: int = Field(1, description="跳帧间隔（1=每帧都处理）", ge=1)
    max_fps: int = Field(30, description="最大处理帧率", ge=1, le=60)
    batch_size: Optional[int] = Field(10, description="批次大小（处理多少帧）", ge=1)
    duration: Optional[int] = Field(None, description="处理时长（秒）", ge=1)


class FrameDetectionResult(BaseModel):
    """单帧检测结果"""
    success: bool
    frame_number: int
    processed_number: int
    detections: List[Detection] = []
    detection_count: int = 0
    inference_time: float = 0
    error: Optional[str] = None


class VideoStreamResponse(BaseModel):
    """视频流检测响应"""
    success: bool
    total_frames: int = Field(..., description="总帧数")
    total_time: float = Field(..., description="总处理时间（秒）")
    avg_fps: float = Field(..., description="平均处理帧率")
    total_detections: int = Field(..., description="总检测数量")
    detections_by_class: Dict[str, int] = Field(default={}, description="各类别检测数量")
    results: List[FrameDetectionResult] = Field(default=[], description="每帧检测结果")
    error: Optional[str] = Field(None, description="错误信息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否加载")
    device: str = Field(..., description="运行设备")
    cuda_available: bool = Field(..., description="CUDA是否可用")
    gpu_name: Optional[str] = Field(None, description="GPU名称")
    model_info: Dict[str, Any] = Field(default={}, description="模型信息")


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    model_path: str
    device: str
    conf_threshold: float
    iou_threshold: float
    max_det: int
    use_half: bool
    class_names: Dict[int, str]
    cuda_available: bool
    gpu_name: Optional[str]
