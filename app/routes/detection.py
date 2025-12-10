"""
检测API路由
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Body
from fastapi.responses import StreamingResponse
from typing import Optional
import cv2
import numpy as np
from loguru import logger
import json

from app.schemas.detection import (
    DetectionRequest,
    DetectionResponse,
    VideoStreamRequest,
    VideoStreamResponse,
    HealthResponse,
    ModelInfoResponse,
    Detection,
    FrameDetectionResult
)
from app.utils.image_utils import encode_image_to_base64
from app.utils.video_processor import VideoStreamProcessor

router = APIRouter(prefix="/api", tags=["detection"])

# 全局检测器实例（在main.py中初始化）
detector = None


def set_detector(det):
    """设置全局检测器实例"""
    global detector
    detector = det


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    model_info = detector.get_model_info()

    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=model_info["device"],
        cuda_available=model_info["cuda_available"],
        gpu_name=model_info.get("gpu_name"),
        model_info=model_info
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    获取模型信息
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    model_info = detector.get_model_info()
    return ModelInfoResponse(**model_info)


@router.post("/detect", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(..., description="图像文件"),
    conf_threshold: Optional[float] = 0.25,
    iou_threshold: Optional[float] = 0.45,
    return_image: bool = False,
    image_format: str = "JPEG"
):
    """
    单图检测接口

    - **file**: 上传的图像文件
    - **conf_threshold**: 置信度阈值（默认0.25）
    - **iou_threshold**: NMS IOU阈值（默认0.45）
    - **return_image**: 是否返回标注后的图像
    - **image_format**: 返回图像格式（JPEG或PNG）
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    try:
        # 读取上传的图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # 执行检测
        result = detector.detect_image(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        # 构建响应
        response_data = {
            "success": result["success"],
            "detections": result["detections"],
            "detection_count": result["detection_count"],
            "inference_time": result["inference_time"],
            "image_shape": list(result["image_shape"])
        }

        # 如果需要返回标注后的图像
        if return_image and result["success"]:
            annotated_image = detector.draw_detections(image, result["detections"])
            image_base64 = encode_image_to_base64(annotated_image, format=image_format)
            response_data["result_image"] = image_base64

        if not result["success"]:
            response_data["error"] = result.get("error", "Unknown error")

        return DetectionResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/batch")
async def detect_batch(
    files: list[UploadFile] = File(..., description="多个图像文件"),
    conf_threshold: Optional[float] = 0.25,
    iou_threshold: Optional[float] = 0.45,
    return_images: bool = False
):
    """
    批量图像检测接口

    - **files**: 多个上传的图像文件
    - **conf_threshold**: 置信度阈值
    - **iou_threshold**: NMS IOU阈值
    - **return_images**: 是否返回标注后的图像
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    results = []

    for file in files:
        try:
            # 读取图像
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid image file"
                })
                continue

            # 执行检测
            result = detector.detect_image(
                image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )

            result_data = {
                "filename": file.filename,
                "success": result["success"],
                "detections": result["detections"],
                "detection_count": result["detection_count"],
                "inference_time": result["inference_time"]
            }

            # 返回标注图像
            if return_images and result["success"]:
                annotated_image = detector.draw_detections(image, result["detections"])
                result_data["result_image"] = encode_image_to_base64(annotated_image)

            if not result["success"]:
                result_data["error"] = result.get("error", "Unknown error")

            results.append(result_data)

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    # 统计信息
    total_detections = sum(r["detection_count"] for r in results if r["success"])
    avg_inference_time = sum(r.get("inference_time", 0) for r in results if r["success"]) / len([r for r in results if r["success"]]) if any(r["success"] for r in results) else 0

    return {
        "success": True,
        "total_images": len(files),
        "successful_detections": sum(1 for r in results if r["success"]),
        "total_detections": total_detections,
        "avg_inference_time": round(avg_inference_time, 4),
        "results": results
    }


@router.post("/detect/video-stream", response_model=VideoStreamResponse)
async def detect_video_stream(request: VideoStreamRequest):
    """
    视频流检测接口

    处理视频流并返回批量检测结果
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    try:
        # 创建视频流处理器
        processor = VideoStreamProcessor(
            stream_url=request.stream_url,
            frame_skip=request.frame_skip,
            max_fps=request.max_fps,
            timeout=30,
            buffer_size=10
        )

        # 定义检测函数
        def detect_func(frame: np.ndarray):
            return detector.detect_image(
                frame,
                conf_threshold=request.conf_threshold,
                iou_threshold=request.iou_threshold
            )

        # 处理视频流
        result = await processor.process_stream_batch(
            detect_func=detect_func,
            batch_size=request.batch_size,
            duration=request.duration
        )

        # 转换结果格式
        frame_results = []
        for r in result.get("results", []):
            frame_result = FrameDetectionResult(
                success=r.get("success", False),
                frame_number=r.get("frame_number", 0),
                processed_number=r.get("processed_number", 0),
                detections=[Detection(**det) for det in r.get("detections", [])],
                detection_count=r.get("detection_count", 0),
                inference_time=r.get("inference_time", 0),
                error=r.get("error")
            )
            frame_results.append(frame_result)

        return VideoStreamResponse(
            success=result["success"],
            total_frames=result["total_frames"],
            total_time=result["total_time"],
            avg_fps=result["avg_fps"],
            total_detections=result["total_detections"],
            detections_by_class=result["detections_by_class"],
            results=frame_results,
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"Video stream detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/video-stream/live")
async def detect_video_stream_live(request: VideoStreamRequest):
    """
    视频流实时检测接口（流式响应）

    返回Server-Sent Events (SSE)流，实时推送检测结果
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    async def event_generator():
        """生成SSE事件"""
        try:
            processor = VideoStreamProcessor(
                stream_url=request.stream_url,
                frame_skip=request.frame_skip,
                max_fps=request.max_fps,
                timeout=30,
                buffer_size=10
            )

            def detect_func(frame: np.ndarray):
                return detector.detect_image(
                    frame,
                    conf_threshold=request.conf_threshold,
                    iou_threshold=request.iou_threshold
                )

            async for result in processor.process_stream(detect_func):
                # 将结果转换为JSON并发送
                event_data = json.dumps(result, ensure_ascii=False)
                yield f"data: {event_data}\n\n"

            # 发送结束事件
            yield "data: {\"event\": \"end\"}\n\n"

        except Exception as e:
            logger.error(f"Live stream error: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
