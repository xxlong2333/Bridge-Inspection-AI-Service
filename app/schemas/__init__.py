from .detection import (
    DetectionRequest,
    DetectionResponse,
    VideoStreamRequest,
    VideoStreamResponse,
    HealthResponse
)

from .traffic import (
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

__all__ = [
    "DetectionRequest",
    "DetectionResponse",
    "VideoStreamRequest",
    "VideoStreamResponse",
    "HealthResponse",
    "TrafficDetectionRequest",
    "TrafficDetectionResponse",
    "TrafficVideoRequest",
    "TrafficVideoResponse",
    "TrafficStreamRequest",
    "TrafficStreamResponse",
    "TrafficHealthResponse",
    "TrafficModelInfoResponse",
    "VehicleDetection",
    "TrafficFrameResult"
]
