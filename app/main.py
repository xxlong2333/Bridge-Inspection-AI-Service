"""
AI检测服务主应用
"""
import os
import yaml
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.models import YOLODetector
from app.routes import detection_router, traffic_router
from app.routes.detection import set_detector
from app.routes.traffic import set_traffic_detector

# 加载配置
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config():
    """加载配置文件"""
    if not CONFIG_PATH.exists():
        logger.warning(f"Config file not found at {CONFIG_PATH}, using defaults")
        return {}

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# 全局配置
config = load_config()

# 全局检测器实例
detector_instance = None
traffic_detector_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global detector_instance, traffic_detector_instance

    # 启动时加载模型
    logger.info("=" * 60)
    logger.info("Starting AI Detection Service")
    logger.info("=" * 60)

    try:
        # 获取可视化颜色配置
        visualization_config = config.get("visualization", {})
        
        # ========== 加载桥梁缺陷检测模型 ==========
        model_config = config.get("model", {})
        weights_path = model_config.get("weights_path", "weights/best.pt")
        device = model_config.get("device", "cuda")
        conf_threshold = model_config.get("conf_threshold", 0.25)
        iou_threshold = model_config.get("iou_threshold", 0.45)
        max_det = model_config.get("max_det", 300)
        use_half = model_config.get("use_half", True)
        class_names = model_config.get("classes", {})
        visualization_colors = visualization_config.get("colors", {})

        # 检查模型文件是否存在
        weights_file = Path(__file__).parent.parent / weights_path
        if not weights_file.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_file}")

        logger.info(f"Loading bridge defect model from {weights_file}")

        # 初始化桥梁缺陷检测器
        detector_instance = YOLODetector(
            model_path=str(weights_file),
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            use_half=use_half,
            class_names=class_names,
            visualization_colors=visualization_colors
        )

        # 设置全局检测器
        set_detector(detector_instance)
        logger.success("Bridge defect model loaded successfully")

        # ========== 加载车流量检测模型 ==========
        traffic_config = config.get("traffic_model", {})
        traffic_weights_path = traffic_config.get("weights_path", "weights/car-best.pt")
        traffic_device = traffic_config.get("device", device)  # 默认使用相同设备
        traffic_conf_threshold = traffic_config.get("conf_threshold", 0.25)
        traffic_iou_threshold = traffic_config.get("iou_threshold", 0.45)
        traffic_max_det = traffic_config.get("max_det", 300)
        traffic_use_half = traffic_config.get("use_half", True)
        traffic_class_names = traffic_config.get("classes", {})
        traffic_visualization_colors = visualization_config.get("traffic_colors", {})

        # 检查车流量模型文件是否存在
        traffic_weights_file = Path(__file__).parent.parent / traffic_weights_path
        if traffic_weights_file.exists():
            logger.info(f"Loading traffic model from {traffic_weights_file}")

            # 初始化车流量检测器
            traffic_detector_instance = YOLODetector(
                model_path=str(traffic_weights_file),
                device=traffic_device,
                conf_threshold=traffic_conf_threshold,
                iou_threshold=traffic_iou_threshold,
                max_det=traffic_max_det,
                use_half=traffic_use_half,
                class_names=traffic_class_names,
                visualization_colors=traffic_visualization_colors
            )

            # 设置全局车流量检测器
            set_traffic_detector(traffic_detector_instance)
            logger.success("Traffic model loaded successfully")
        else:
            logger.warning(f"Traffic model weights not found at {traffic_weights_file}, traffic detection disabled")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # 关闭时清理资源
    logger.info("Shutting down AI Detection Service")


# 创建FastAPI应用
app = FastAPI(
    title="Bridge Inspection AI Service",
    description="YOLOv8桥梁缺陷检测服务 - 支持裂缝、混凝土剥落、钢索锈蚀检测，以及车流量检测",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "message": "Internal server error"
        }
    )


# 注册路由
app.include_router(detection_router)
app.include_router(traffic_router)


# 根路由
@app.get("/")
async def root():
    """根路由"""
    return {
        "service": "Bridge Inspection AI Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "bridge_detection": {
                "health": "/api/health",
                "model_info": "/api/model/info",
                "detect_image": "/api/detect",
                "detect_batch": "/api/detect/batch",
                "detect_video_stream": "/api/detect/video-stream",
                "detect_video_stream_live": "/api/detect/video-stream/live"
            },
            "traffic_detection": {
                "health": "/api/traffic/health",
                "model_info": "/api/traffic/model/info",
                "detect_image": "/api/traffic/detect",
                "detect_batch": "/api/traffic/detect/batch",
                "detect_video": "/api/traffic/detect/video",
                "detect_stream": "/api/traffic/stream",
                "detect_stream_live": "/api/traffic/stream/live"
            },
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn

    # 获取服务器配置
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 5000)
    workers = server_config.get("workers", 1)
    reload = server_config.get("reload", False)
    log_level = server_config.get("log_level", "info")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level
    )
