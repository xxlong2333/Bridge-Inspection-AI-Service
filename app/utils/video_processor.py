"""
视频流处理器
支持从URL获取视频流并进行实时检测
"""
import asyncio
import time
from typing import Optional, AsyncGenerator, Dict, Any, Callable
import cv2
import numpy as np
from loguru import logger
import aiohttp


class VideoStreamProcessor:
    """视频流处理器"""

    def __init__(
        self,
        stream_url: str,
        frame_skip: int = 1,
        max_fps: int = 30,
        timeout: int = 30,
        buffer_size: int = 10
    ):
        """
        初始化视频流处理器

        Args:
            stream_url: 视频流URL
            frame_skip: 跳帧间隔（1=每帧都处理）
            max_fps: 最大处理帧率
            timeout: 连接超时时间（秒）
            buffer_size: 帧缓冲区大小
        """
        self.stream_url = stream_url
        self.frame_skip = frame_skip
        self.max_fps = max_fps
        self.timeout = timeout
        self.buffer_size = buffer_size

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_count = 0
        self.processed_count = 0

        logger.info(f"VideoStreamProcessor initialized for {stream_url}")

    async def open_stream(self) -> bool:
        """
        打开视频流

        Returns:
            是否成功打开
        """
        try:
            logger.info(f"Opening video stream: {self.stream_url}")

            # 尝试打开视频流
            self.cap = cv2.VideoCapture(self.stream_url)

            if not self.cap.isOpened():
                logger.error("Failed to open video stream")
                return False

            # 设置缓冲区
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            self.is_running = True
            logger.success("Video stream opened successfully")

            # 获取流信息
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Stream info - FPS: {fps}, Size: {width}x{height}")

            return True

        except Exception as e:
            logger.error(f"Error opening stream: {e}")
            return False

    def close_stream(self):
        """关闭视频流"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.is_running = False
        logger.info("Video stream closed")

    async def read_frame(self) -> Optional[np.ndarray]:
        """
        读取一帧

        Returns:
            视频帧 或 None（如果读取失败）
        """
        if not self.is_running or self.cap is None:
            return None

        try:
            # 异步读取帧
            ret, frame = await asyncio.to_thread(self.cap.read)

            if not ret:
                logger.warning("Failed to read frame")
                return None

            self.frame_count += 1
            return frame

        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None

    async def process_stream(
        self,
        detect_func: Callable[[np.ndarray], Dict[str, Any]],
        on_frame: Optional[Callable[[np.ndarray, Dict[str, Any]], None]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        处理视频流（生成器模式）

        Args:
            detect_func: 检测函数，接收帧并返回检测结果
            on_frame: 帧回调函数（可选）

        Yields:
            检测结果字典
        """
        if not await self.open_stream():
            raise RuntimeError("Failed to open video stream")

        frame_interval = 1.0 / self.max_fps
        last_process_time = 0

        try:
            while self.is_running:
                # 读取帧
                frame = await self.read_frame()
                if frame is None:
                    break

                # 跳帧处理
                if self.frame_count % self.frame_skip != 0:
                    continue

                # 帧率控制
                current_time = time.time()
                elapsed = current_time - last_process_time
                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)

                last_process_time = time.time()

                # 执行检测
                try:
                    result = await asyncio.to_thread(detect_func, frame)
                    result["frame_number"] = self.frame_count
                    result["processed_number"] = self.processed_count

                    self.processed_count += 1

                    # 回调
                    if on_frame:
                        await asyncio.to_thread(on_frame, frame, result)

                    yield result

                except Exception as e:
                    logger.error(f"Detection error on frame {self.frame_count}: {e}")
                    yield {
                        "success": False,
                        "error": str(e),
                        "frame_number": self.frame_count
                    }

        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            raise

        finally:
            self.close_stream()

    async def process_stream_batch(
        self,
        detect_func: Callable[[np.ndarray], Dict[str, Any]],
        batch_size: int = 10,
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        批量处理视频流（处理固定数量帧后返回）

        Args:
            detect_func: 检测函数
            batch_size: 批次大小（处理多少帧）
            duration: 处理时长（秒），优先于batch_size

        Returns:
            批次处理结果
        """
        if not await self.open_stream():
            raise RuntimeError("Failed to open video stream")

        start_time = time.time()
        results = []
        frame_times = []

        try:
            async for result in self.process_stream(detect_func):
                results.append(result)
                frame_times.append(time.time())

                # 检查停止条件
                if duration and (time.time() - start_time) >= duration:
                    break
                if not duration and len(results) >= batch_size:
                    break

            # 计算统计信息
            total_time = time.time() - start_time
            avg_fps = len(results) / total_time if total_time > 0 else 0

            # 统计检测结果
            total_detections = sum(r.get("detection_count", 0) for r in results if r.get("success"))
            detection_counts_by_class = {}

            for result in results:
                if not result.get("success"):
                    continue
                for det in result.get("detections", []):
                    class_name = det["class_name"]
                    detection_counts_by_class[class_name] = detection_counts_by_class.get(class_name, 0) + 1

            return {
                "success": True,
                "total_frames": len(results),
                "total_time": round(total_time, 2),
                "avg_fps": round(avg_fps, 2),
                "total_detections": total_detections,
                "detections_by_class": detection_counts_by_class,
                "results": results
            }

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_frames": len(results),
                "results": results
            }

        finally:
            self.close_stream()

    def get_stream_info(self) -> Dict[str, Any]:
        """获取视频流信息"""
        if self.cap is None or not self.cap.isOpened():
            return {"is_opened": False}

        return {
            "is_opened": True,
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": self.frame_count,
            "processed_count": self.processed_count
        }


async def download_video_stream(
    url: str,
    timeout: int = 30
) -> Optional[bytes]:
    """
    下载视频流数据（用于小视频文件）

    Args:
        url: 视频URL
        timeout: 超时时间

    Returns:
        视频字节数据 或 None
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Failed to download video: HTTP {response.status}")
                    return None

    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return None
