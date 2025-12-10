"""
车流量检测API路由
支持实时视频流检测、车辆跟踪和流式进度反馈
"""
import os
import time
import tempfile
import uuid
import asyncio
import subprocess
import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse
from typing import Optional
import cv2
import numpy as np
from loguru import logger
import json

from app.schemas.traffic import (
    TrafficDetectionRequest,
    TrafficDetectionResponse,
    TrafficVideoRequest,
    TrafficVideoResponse,
    TrafficStreamRequest,
    TrafficStreamResponse,
    TrafficHealthResponse,
    TrafficModelInfoResponse,
    VehicleDetection,
    TrafficFrameResult
)
from app.utils.image_utils import encode_image_to_base64
from app.utils.video_processor import VideoStreamProcessor
from app.models import VehicleTracker

router = APIRouter(prefix="/api/traffic", tags=["traffic"])

# 全局车辆检测器实例（在main.py中初始化）
traffic_detector = None

# 临时文件目录
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)


def get_video_writer(output_path: str, fps: float, width: int, height: int):
    """
    获取视频写入器，尝试使用浏览器兼容的编码器
    
    优先级：H264 > avc1 > XVID > mp4v
    """
    # 尝试不同的编码器
    codecs_to_try = [
        ('avc1', 'mp4'),   # H.264 for MP4 container
        ('H264', 'mp4'),   # H.264 alternative
        ('X264', 'mp4'),   # x264 encoder
        ('XVID', 'avi'),   # Xvid (fallback, change extension)
        ('mp4v', 'mp4'),   # MPEG-4 Part 2 (last resort)
    ]
    
    for codec, ext in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            # 如果需要更改扩展名
            if ext != 'mp4':
                output_path = output_path.rsplit('.', 1)[0] + f'.{ext}'
            
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if writer.isOpened():
                logger.info(f"Using video codec: {codec}")
                return writer, output_path
            else:
                writer.release()
        except Exception as e:
            logger.debug(f"Codec {codec} not available: {e}")
            continue
    
    # 如果所有编码器都失败，使用默认的
    logger.warning("No browser-compatible codec found, using mp4v")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height)), output_path


def convert_to_h264(input_path: str, output_path: str) -> bool:
    """
    使用ffmpeg将视频转换为H.264编码（浏览器兼容）
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        
    Returns:
        是否成功转换
    """
    try:
        # 检查ffmpeg是否可用
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            logger.warning("ffmpeg not found, skipping H.264 conversion")
            return False
        
        # 使用ffmpeg转换为H.264
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',  # 优化网络播放
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"Successfully converted video to H.264: {output_path}")
            return True
        else:
            logger.error(f"ffmpeg conversion failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg conversion timeout")
        return False
    except Exception as e:
        logger.error(f"Video conversion error: {e}")
        return False


def set_traffic_detector(det):
    """设置全局车辆检测器实例"""
    global traffic_detector
    traffic_detector = det


@router.get("/health", response_model=TrafficHealthResponse)
async def health_check():
    """
    车流量检测服务健康检查接口
    """
    if traffic_detector is None:
        raise HTTPException(status_code=503, detail="Traffic detector not initialized")

    model_info = traffic_detector.get_model_info()

    return TrafficHealthResponse(
        status="ok",
        model_loaded=True,
        device=model_info["device"],
        cuda_available=model_info["cuda_available"],
        gpu_name=model_info.get("gpu_name"),
        model_info=model_info
    )


@router.get("/model/info", response_model=TrafficModelInfoResponse)
async def get_model_info():
    """
    获取车流量检测模型信息
    """
    if traffic_detector is None:
        raise HTTPException(status_code=503, detail="Traffic detector not initialized")

    model_info = traffic_detector.get_model_info()
    return TrafficModelInfoResponse(**model_info)


def _count_vehicles_by_type(detections: list) -> dict:
    """统计各类型车辆数量"""
    count_by_type = {}
    for det in detections:
        class_name = det.get("class_name", "unknown")
        count_by_type[class_name] = count_by_type.get(class_name, 0) + 1
    return count_by_type


@router.post("/detect", response_model=TrafficDetectionResponse)
async def detect_vehicles(
    file: UploadFile = File(..., description="图像文件"),
    conf_threshold: Optional[float] = Form(0.25),
    iou_threshold: Optional[float] = Form(0.45),
    return_image: bool = Form(False),
    image_format: str = Form("JPEG")
):
    """
    车辆检测接口（单图）

    - **file**: 上传的图像文件
    - **conf_threshold**: 置信度阈值（默认0.25）
    - **iou_threshold**: NMS IOU阈值（默认0.45）
    - **return_image**: 是否返回标注后的图像
    - **image_format**: 返回图像格式（JPEG或PNG）
    """
    if traffic_detector is None:
        raise HTTPException(status_code=503, detail="Traffic detector not initialized")

    try:
        # 读取上传的图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # 执行检测
        result = traffic_detector.detect_image(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        # 统计各类型车辆数量
        vehicle_count_by_type = _count_vehicles_by_type(result["detections"])

        # 转换检测结果格式
        detections = [
            VehicleDetection(
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                bbox=det["bbox"],
                bbox_center=det["bbox_center"],
                bbox_area=det["bbox_area"]
            )
            for det in result["detections"]
        ]

        # 构建响应
        response_data = {
            "success": result["success"],
            "detections": detections,
            "vehicle_count": result["detection_count"],
            "vehicle_count_by_type": vehicle_count_by_type,
            "inference_time": result["inference_time"],
            "image_shape": list(result["image_shape"]) if result.get("image_shape") else None
        }

        # 如果需要返回标注后的图像
        if return_image and result["success"]:
            annotated_image = traffic_detector.draw_detections(image, result["detections"])
            image_base64 = encode_image_to_base64(annotated_image, format=image_format)
            response_data["result_image"] = image_base64

        if not result["success"]:
            response_data["error"] = result.get("error", "Unknown error")

        return TrafficDetectionResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vehicle detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/batch")
async def detect_vehicles_batch(
    files: list[UploadFile] = File(..., description="多个图像文件"),
    conf_threshold: Optional[float] = Form(0.25),
    iou_threshold: Optional[float] = Form(0.45),
    return_images: bool = Form(False)
):
    """
    批量车辆检测接口

    - **files**: 多个上传的图像文件
    - **conf_threshold**: 置信度阈值
    - **iou_threshold**: NMS IOU阈值
    - **return_images**: 是否返回标注后的图像
    """
    if traffic_detector is None:
        raise HTTPException(status_code=503, detail="Traffic detector not initialized")

    results = []
    total_vehicle_count = 0
    total_by_type = {}

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
            result = traffic_detector.detect_image(
                image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )

            vehicle_count_by_type = _count_vehicles_by_type(result["detections"])

            result_data = {
                "filename": file.filename,
                "success": result["success"],
                "detections": result["detections"],
                "vehicle_count": result["detection_count"],
                "vehicle_count_by_type": vehicle_count_by_type,
                "inference_time": result["inference_time"]
            }

            # 累计统计
            if result["success"]:
                total_vehicle_count += result["detection_count"]
                for vtype, count in vehicle_count_by_type.items():
                    total_by_type[vtype] = total_by_type.get(vtype, 0) + count

            # 返回标注图像
            if return_images and result["success"]:
                annotated_image = traffic_detector.draw_detections(image, result["detections"])
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
    successful_count = sum(1 for r in results if r["success"])
    avg_inference_time = (
        sum(r.get("inference_time", 0) for r in results if r["success"]) / successful_count
        if successful_count > 0 else 0
    )

    return {
        "success": True,
        "total_images": len(files),
        "successful_detections": successful_count,
        "total_vehicle_count": total_vehicle_count,
        "total_by_type": total_by_type,
        "avg_inference_time": round(avg_inference_time, 4),
        "results": results
    }


@router.post("/detect/video", response_model=TrafficVideoResponse)
async def detect_vehicles_video(
    file: UploadFile = File(..., description="视频文件"),
    conf_threshold: Optional[float] = Form(0.25),
    iou_threshold: Optional[float] = Form(0.45),
    frame_skip: int = Form(1),
    return_video: bool = Form(False),
    enable_tracking: bool = Form(True)
):
    """
    车流量视频检测接口（同步模式）

    - **file**: 上传的视频文件
    - **conf_threshold**: 置信度阈值（默认0.25）
    - **iou_threshold**: NMS IOU阈值（默认0.45）
    - **frame_skip**: 跳帧间隔（1=每帧都处理）
    - **return_video**: 是否返回处理后的视频
    - **enable_tracking**: 是否启用车辆跟踪（去重计数）
    """
    if traffic_detector is None:
        raise HTTPException(status_code=503, detail="Traffic detector not initialized")

    # 保存上传的视频到临时文件
    video_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"input_{video_id}.mp4")
    output_path_temp = os.path.join(TEMP_DIR, f"output_temp_{video_id}.mp4")
    output_path = os.path.join(TEMP_DIR, f"output_{video_id}.mp4")

    try:
        # 保存上传的视频
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)

        # 打开视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")

        # 获取视频参数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 如果需要输出视频，尝试使用浏览器兼容的编码器
        out = None
        actual_output_path = output_path_temp
        if return_video:
            out, actual_output_path = get_video_writer(output_path_temp, fps, width, height)

        # 初始化车辆跟踪器
        vehicle_tracker = VehicleTracker() if enable_tracking else None
        if enable_tracking:
            traffic_detector.reset_tracker()

        start_time = time.time()
        frame_count = 0
        processed_count = 0
        total_vehicle_count = 0
        unique_vehicle_count = 0
        vehicle_count_by_type = {}
        frame_vehicle_counts = []
        
        # 保存上一次的检测结果，用于跳帧时绘制标注
        last_detections = []
        last_detection_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 跳帧处理：使用上一次的检测结果绘制标注
            if frame_count % frame_skip != 0:
                if out:
                    # 使用上一次的检测结果绘制，保持标注连续
                    if last_detections:
                        annotated_frame = traffic_detector.draw_detections(frame, last_detections)
                        count_text = f"Unique: {unique_vehicle_count}" if enable_tracking else f"Vehicles: {last_detection_count}"
                        cv2.putText(annotated_frame, count_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Current: {last_detection_count}", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        out.write(annotated_frame)
                    else:
                        out.write(frame)
                continue

            processed_count += 1

            # 执行检测（带跟踪或不带跟踪）
            if enable_tracking:
                result = traffic_detector.track_video_frame(
                    frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    persist=True
                )
            else:
                result = traffic_detector.detect_image(
                    frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )

            if result["success"]:
                frame_vehicle_counts.append(result["detection_count"])
                total_vehicle_count += result["detection_count"]
                
                # 保存当前检测结果用于下一帧
                last_detections = result["detections"]
                last_detection_count = result["detection_count"]

                # 更新跟踪统计
                if enable_tracking and vehicle_tracker:
                    new_count, new_by_type = vehicle_tracker.update(result["detections"])
                    unique_vehicle_count = vehicle_tracker.total_count
                    vehicle_count_by_type = vehicle_tracker.vehicle_counts.copy()
                else:
                    # 不跟踪时直接累加
                    for det in result["detections"]:
                        class_name = det.get("class_name", "unknown")
                        vehicle_count_by_type[class_name] = vehicle_count_by_type.get(class_name, 0) + 1

                # 绘制检测结果
                if out:
                    annotated_frame = traffic_detector.draw_detections(frame, result["detections"])
                    # 添加车辆计数信息
                    count_text = f"Unique: {unique_vehicle_count}" if enable_tracking else f"Vehicles: {result['detection_count']}"
                    cv2.putText(
                        annotated_frame,
                        count_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    # 显示当前帧车辆数
                    cv2.putText(
                        annotated_frame,
                        f"Current: {result['detection_count']}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )
                    out.write(annotated_frame)
            elif out:
                # 检测失败时也使用上一次的结果
                if last_detections:
                    annotated_frame = traffic_detector.draw_detections(frame, last_detections)
                    count_text = f"Unique: {unique_vehicle_count}" if enable_tracking else f"Vehicles: {last_detection_count}"
                    cv2.putText(annotated_frame, count_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    out.write(annotated_frame)
                else:
                    out.write(frame)

        cap.release()
        if out:
            out.release()

        # 尝试转换为H.264格式（浏览器兼容）
        if return_video and os.path.exists(actual_output_path):
            h264_converted = convert_to_h264(actual_output_path, output_path)
            if not h264_converted:
                # 如果转换失败，使用原始文件
                if actual_output_path != output_path:
                    shutil.move(actual_output_path, output_path)
            else:
                if actual_output_path != output_path and os.path.exists(actual_output_path):
                    os.remove(actual_output_path)

        total_time = time.time() - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0
        avg_vehicles_per_frame = (
            sum(frame_vehicle_counts) / len(frame_vehicle_counts)
            if frame_vehicle_counts else 0
        )

        response_data = {
            "success": True,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "total_vehicle_count": unique_vehicle_count if enable_tracking else total_vehicle_count,
            "vehicle_count_by_type": vehicle_count_by_type,
            "avg_vehicles_per_frame": round(avg_vehicles_per_frame, 2),
            "total_time": round(total_time, 2),
            "avg_fps": round(avg_fps, 2)
        }

        if return_video and os.path.exists(output_path):
            response_data["result_video_path"] = f"/api/traffic/video/{video_id}"

        return TrafficVideoResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理输入临时文件
        if os.path.exists(input_path):
            os.remove(input_path)
        # 清理临时输出文件
        for temp_path in [output_path_temp, actual_output_path]:
            if temp_path and temp_path != output_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass


@router.post("/detect/video/stream")
async def detect_vehicles_video_stream(
    file: UploadFile = File(..., description="视频文件"),
    conf_threshold: Optional[float] = Form(0.25),
    iou_threshold: Optional[float] = Form(0.45),
    frame_skip: int = Form(1),
    return_video: bool = Form(True),
    enable_tracking: bool = Form(True)
):
    """
    车流量视频流式检测接口（SSE实时进度推送）
    
    返回Server-Sent Events流，实时推送处理进度和检测结果
    
    - **file**: 上传的视频文件
    - **conf_threshold**: 置信度阈值
    - **iou_threshold**: NMS IOU阈值
    - **frame_skip**: 跳帧间隔
    - **return_video**: 是否返回处理后的视频
    - **enable_tracking**: 是否启用车辆跟踪
    """
    if traffic_detector is None:
        raise HTTPException(status_code=503, detail="Traffic detector not initialized")
    
    # 保存上传的视频到临时文件
    video_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"input_{video_id}.mp4")
    output_path_temp = os.path.join(TEMP_DIR, f"output_temp_{video_id}.mp4")
    output_path = os.path.join(TEMP_DIR, f"output_{video_id}.mp4")
    
    # 保存上传的视频
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)
    
    async def event_generator():
        """生成SSE事件"""
        cap = None
        out = None
        actual_output_path = output_path_temp
        
        try:
            # 打开视频
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                yield f"data: {json.dumps({'event': 'error', 'error': 'Invalid video file'})}\n\n"
                return
            
            # 获取视频参数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 发送初始化事件
            init_data = {
                'event': 'init',
                'video_id': video_id,
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height
            }
            yield f"data: {json.dumps(init_data)}\n\n"
            
            # 初始化视频写入器，尝试使用浏览器兼容的编码器
            if return_video:
                out, actual_output_path = get_video_writer(output_path_temp, fps, width, height)
            
            # 初始化车辆跟踪器
            vehicle_tracker = VehicleTracker() if enable_tracking else None
            if enable_tracking:
                traffic_detector.reset_tracker()
            
            start_time = time.time()
            frame_count = 0
            processed_count = 0
            total_vehicle_count = 0
            unique_vehicle_count = 0
            vehicle_count_by_type = {}
            frame_results = []
            
            # 保存上一次的检测结果，用于跳帧时绘制标注
            last_detections = []
            last_detection_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 跳帧处理：使用上一次的检测结果绘制标注
                if frame_count % frame_skip != 0:
                    if out:
                        # 使用上一次的检测结果绘制，保持标注连续
                        if last_detections:
                            annotated_frame = traffic_detector.draw_detections(frame, last_detections)
                            count_text = f"Unique: {unique_vehicle_count}" if enable_tracking else f"Vehicles: {last_detection_count}"
                            cv2.putText(annotated_frame, count_text, (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"Current: {last_detection_count}", (10, 70),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                            out.write(annotated_frame)
                        else:
                            out.write(frame)
                    continue
                
                processed_count += 1
                
                # 执行检测
                if enable_tracking:
                    result = traffic_detector.track_video_frame(
                        frame,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        persist=True
                    )
                else:
                    result = traffic_detector.detect_image(
                        frame,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold
                    )
                
                current_count = 0
                if result["success"]:
                    current_count = result["detection_count"]
                    total_vehicle_count += current_count
                    
                    # 保存当前检测结果用于下一帧
                    last_detections = result["detections"]
                    last_detection_count = current_count
                    
                    # 更新跟踪统计
                    if enable_tracking and vehicle_tracker:
                        new_count, _ = vehicle_tracker.update(result["detections"])
                        unique_vehicle_count = vehicle_tracker.total_count
                        vehicle_count_by_type = vehicle_tracker.vehicle_counts.copy()
                    else:
                        for det in result["detections"]:
                            class_name = det.get("class_name", "unknown")
                            vehicle_count_by_type[class_name] = vehicle_count_by_type.get(class_name, 0) + 1
                    
                    # 绘制检测结果
                    if out:
                        annotated_frame = traffic_detector.draw_detections(frame, result["detections"])
                        count_text = f"Unique: {unique_vehicle_count}" if enable_tracking else f"Vehicles: {current_count}"
                        cv2.putText(annotated_frame, count_text, (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Current: {current_count}", (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        out.write(annotated_frame)
                    
                    # 保存帧结果
                    frame_results.append({
                        'frame_number': frame_count,
                        'vehicle_count': current_count,
                        'timestamp': frame_count / fps if fps > 0 else 0
                    })
                elif out:
                    # 检测失败时也使用上一次的结果
                    if last_detections:
                        annotated_frame = traffic_detector.draw_detections(frame, last_detections)
                        count_text = f"Unique: {unique_vehicle_count}" if enable_tracking else f"Vehicles: {last_detection_count}"
                        cv2.putText(annotated_frame, count_text, (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        out.write(annotated_frame)
                    else:
                        out.write(frame)
                
                # 发送进度事件（每10帧发送一次）
                if processed_count % 10 == 0 or frame_count == total_frames:
                    progress = round(frame_count / total_frames * 100, 1) if total_frames > 0 else 0
                    progress_data = {
                        'event': 'progress',
                        'frame': frame_count,
                        'total_frames': total_frames,
                        'progress': progress,
                        'processed': processed_count,
                        'current_count': current_count,
                        'unique_count': unique_vehicle_count if enable_tracking else total_vehicle_count,
                        'vehicle_count_by_type': vehicle_count_by_type,
                        'fps': round(processed_count / (time.time() - start_time), 1) if time.time() > start_time else 0
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    await asyncio.sleep(0)  # 让出控制权
            
            cap.release()
            if out:
                out.release()
            
            # 转换为H.264格式
            video_path = None
            if return_video and os.path.exists(actual_output_path):
                h264_converted = convert_to_h264(actual_output_path, output_path)
                if not h264_converted:
                    # 如果转换失败，使用原始文件
                    if actual_output_path != output_path:
                        shutil.move(actual_output_path, output_path)
                else:
                    if actual_output_path != output_path and os.path.exists(actual_output_path):
                        try:
                            os.remove(actual_output_path)
                        except:
                            pass
                video_path = f"/api/traffic/video/{video_id}"
            
            # 发送完成事件
            total_time = time.time() - start_time
            complete_data = {
                'event': 'complete',
                'video_id': video_id,
                'total_frames': total_frames,
                'processed_frames': processed_count,
                'total_vehicle_count': unique_vehicle_count if enable_tracking else total_vehicle_count,
                'vehicle_count_by_type': vehicle_count_by_type,
                'avg_vehicles_per_frame': round(total_vehicle_count / processed_count, 2) if processed_count > 0 else 0,
                'total_time': round(total_time, 2),
                'avg_fps': round(processed_count / total_time, 2) if total_time > 0 else 0,
                'result_video_path': video_path,
                'frame_results': frame_results
            }
            yield f"data: {json.dumps(complete_data)}\n\n"
            
        except Exception as e:
            logger.error(f"Video stream detection error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
        
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            # 清理输入文件
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except:
                    pass
            # 清理临时输出文件
            for temp_path in [output_path_temp, actual_output_path]:
                if temp_path and temp_path != output_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/video/{video_id}")
async def get_processed_video(video_id: str):
    """
    获取处理后的视频文件
    """
    output_path = os.path.join(TEMP_DIR, f"output_{video_id}.mp4")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"traffic_detection_{video_id}.mp4"
    )


@router.post("/stream", response_model=TrafficStreamResponse)
async def detect_vehicles_stream(request: TrafficStreamRequest):
    """
    车流量视频流检测接口

    处理视频流并返回批量检测结果
    """
    if traffic_detector is None:
        raise HTTPException(status_code=503, detail="Traffic detector not initialized")

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
            return traffic_detector.detect_image(
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

        # 统计车流量
        total_vehicle_count = 0
        vehicle_count_by_type = {}
        frame_vehicle_counts = []

        frame_results = []
        for r in result.get("results", []):
            if r.get("success", False):
                detection_count = r.get("detection_count", 0)
                total_vehicle_count += detection_count
                frame_vehicle_counts.append(detection_count)

                for det in r.get("detections", []):
                    class_name = det.get("class_name", "unknown")
                    vehicle_count_by_type[class_name] = vehicle_count_by_type.get(class_name, 0) + 1

            frame_result = TrafficFrameResult(
                success=r.get("success", False),
                frame_number=r.get("frame_number", 0),
                processed_number=r.get("processed_number", 0),
                detections=[
                    VehicleDetection(
                        class_id=det["class_id"],
                        class_name=det["class_name"],
                        confidence=det["confidence"],
                        bbox=det["bbox"],
                        bbox_center=det["bbox_center"],
                        bbox_area=det["bbox_area"]
                    )
                    for det in r.get("detections", [])
                ],
                vehicle_count=r.get("detection_count", 0),
                inference_time=r.get("inference_time", 0),
                error=r.get("error")
            )
            frame_results.append(frame_result)

        avg_vehicles_per_frame = (
            sum(frame_vehicle_counts) / len(frame_vehicle_counts)
            if frame_vehicle_counts else 0
        )

        return TrafficStreamResponse(
            success=result["success"],
            total_frames=result["total_frames"],
            total_time=result["total_time"],
            avg_fps=result["avg_fps"],
            total_vehicle_count=total_vehicle_count,
            vehicle_count_by_type=vehicle_count_by_type,
            avg_vehicles_per_frame=round(avg_vehicles_per_frame, 2),
            results=frame_results,
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"Video stream detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream/live")
async def detect_vehicles_stream_live(request: TrafficStreamRequest):
    """
    车流量视频流实时检测接口（流式响应）

    返回Server-Sent Events (SSE)流，实时推送检测结果
    """
    if traffic_detector is None:
        raise HTTPException(status_code=503, detail="Traffic detector not initialized")

    async def event_generator():
        """生成SSE事件"""
        total_vehicle_count = 0
        vehicle_count_by_type = {}

        try:
            processor = VideoStreamProcessor(
                stream_url=request.stream_url,
                frame_skip=request.frame_skip,
                max_fps=request.max_fps,
                timeout=30,
                buffer_size=10
            )

            def detect_func(frame: np.ndarray):
                return traffic_detector.detect_image(
                    frame,
                    conf_threshold=request.conf_threshold,
                    iou_threshold=request.iou_threshold
                )

            async for result in processor.process_stream(detect_func):
                # 更新累计统计
                if result.get("success", False):
                    total_vehicle_count += result.get("detection_count", 0)
                    for det in result.get("detections", []):
                        class_name = det.get("class_name", "unknown")
                        vehicle_count_by_type[class_name] = vehicle_count_by_type.get(class_name, 0) + 1

                # 添加累计统计到结果
                result["total_vehicle_count"] = total_vehicle_count
                result["vehicle_count_by_type"] = vehicle_count_by_type

                # 将结果转换为JSON并发送
                event_data = json.dumps(result, ensure_ascii=False)
                yield f"data: {event_data}\n\n"

            # 发送结束事件
            end_data = json.dumps({
                "event": "end",
                "total_vehicle_count": total_vehicle_count,
                "vehicle_count_by_type": vehicle_count_by_type
            })
            yield f"data: {end_data}\n\n"

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



