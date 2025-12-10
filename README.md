# Bridge Inspection AI Service

基于 YOLOv8 的桥梁缺陷检测及车流检测 AI 服务，支持实时视频流检测。

## 功能特性

- ✅ **单图检测**：上传图片进行缺陷检测
- ✅ **批量检测**：一次上传多张图片批量检测
- ✅ **视频流检测**：从 API 获取视频流并实时检测
- ✅ **实时流式检测**：SSE 流式推送检测结果
- ✅ **三种缺陷类型**：裂缝、混凝土剥落、钢索锈蚀

## 技术栈

- **框架**: FastAPI
- **深度学习**: YOLOv8 (Ultralytics)
- **图像处理**: OpenCV, Pillow
- **异步支持**: asyncio, aiohttp

## 项目结构

```
ai-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # 主应用入口
│   ├── models/
│   │   ├── __init__.py
│   │   └── yolo_detector.py # YOLO检测器封装
│   ├── routes/
│   │   ├── __init__.py
│   │   └── detection.py     # 检测API路由
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── detection.py     # 请求/响应模型
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py   # 图像处理工具
│       └── video_processor.py # 视频流处理
├── weights/
│   └── best.pt              # 训练好的模型权重
├── config.yaml              # 配置文件
├── requirements.txt         # Python依赖
├── Dockerfile              # Docker镜像配置
├── docker-compose.yml      # Docker Compose配置
├── start.sh                # Linux/Mac启动脚本
├── start.bat               # Windows启动脚本
└── README.md               # 本文件
```

## 快速开始

### Windows 快捷脚本

直接执行 start-win.bat

### 方式 1：本地运行

#### 1. 环境要求

- Python 3.10+
- CUDA 11.8+ (可选，用于 GPU 加速)

#### 2. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 3. 配置模型

确保模型权重文件位于 `weights/best.pt`
确保模型权重文件位于 `weights/car-best.pt`

#### 4. 修改配置（可选）

编辑 `config.yaml` 调整模型参数、类别名称等配置。

#### 5. 启动服务

**Windows:**

```bash
start.bat
```

**Linux/Mac:**

```bash
chmod +x start.sh
./start.sh
```

**或直接运行:**

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000
```

服务启动后访问：

- API 文档: http://localhost:5000/docs
- ReDoc 文档: http://localhost:5000/redoc

### 方式 2：Docker 运行

#### 1. 使用 Docker Compose（推荐）

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

#### 2. 使用 Docker 命令

```bash
# 构建镜像
docker build -t bridge-ai-service .

# 运行容器
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/config.yaml:/app/config.yaml \
  --name bridge-ai-service \
  bridge-ai-service

# 查看日志
docker logs -f bridge-ai-service
```

#### GPU 支持（可选）

如果有 NVIDIA GPU，修改 `docker-compose.yml` 取消 GPU 配置的注释，并确保安装了 nvidia-docker。

## API 接口说明

### 1. 健康检查

```http
GET /api/health
```

**响应示例:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060"
}
```

### 2. 获取模型信息

```http
GET /api/model/info
```

### 3. 单图检测

```http
POST /api/detect
Content-Type: multipart/form-data

Parameters:
  - file: 图像文件
  - conf_threshold: 置信度阈值 (默认0.25)
  - iou_threshold: NMS阈值 (默认0.45)
  - return_image: 是否返回标注图像 (默认false)
  - image_format: 返回图像格式 (JPEG/PNG)
```

**响应示例:**

```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "crack",
      "confidence": 0.89,
      "bbox": [100.5, 200.3, 150.2, 250.8],
      "bbox_center": [125.35, 225.55],
      "bbox_area": 2512.5
    }
  ],
  "detection_count": 1,
  "inference_time": 0.045,
  "image_shape": [1080, 1920]
}
```

### 4. 批量检测

```http
POST /api/detect/batch
Content-Type: multipart/form-data

Parameters:
  - files: 多个图像文件
  - conf_threshold: 置信度阈值
  - iou_threshold: NMS阈值
  - return_images: 是否返回标注图像
```

### 5. 视频流检测（批量模式）

```http
POST /api/detect/video-stream
Content-Type: application/json

Body:
{
  "stream_url": "http://example.com/video/stream",
  "conf_threshold": 0.25,
  "iou_threshold": 0.45,
  "frame_skip": 1,
  "max_fps": 30,
  "batch_size": 10,
  "duration": null
}
```

**响应示例:**

```json
{
  "success": true,
  "total_frames": 10,
  "total_time": 2.5,
  "avg_fps": 4.0,
  "total_detections": 15,
  "detections_by_class": {
    "crack": 8,
    "concrete_spalling": 5,
    "cable_corrosion": 2
  },
  "results": [...]
}
```

### 6. 视频流实时检测（SSE 流式）

```http
POST /api/detect/video-stream/live
Content-Type: application/json

Body:
{
  "stream_url": "http://example.com/video/stream",
  "conf_threshold": 0.25,
  "frame_skip": 2,
  "max_fps": 15
}
```

返回 Server-Sent Events (SSE)流，实时推送每帧检测结果。

## 配置说明

### config.yaml

```yaml
# 服务配置
server:
  host: "0.0.0.0"
  port: 5000

# 模型配置
model:
  weights_path: "weights/best.pt"
  device: "cuda" # cuda, cpu, mps
  conf_threshold: 0.25
  iou_threshold: 0.45

  # 类别名称（根据训练数据修改）
  classes:
    0: "crack"
    1: "concrete_spalling"
    2: "cable_corrosion"

# 视频流配置
video:
  frame_skip: 1
  max_fps: 30
  timeout: 30
```

## 性能优化建议

1. **GPU 加速**：使用 CUDA 可大幅提升性能

   - CPU: ~200-500ms/张
   - GPU: ~20-50ms/张

2. **半精度推理**：在 config.yaml 中设置 `use_half: true`（仅 GPU 支持）

3. **跳帧处理**：视频流检测时设置 `frame_skip > 1` 降低计算量

4. **批量推理**：多图检测时一次性处理，提高吞吐量

## 常见问题

### 1. 模型加载失败

确保 `weights/best.pt` 文件存在且是有效的 YOLOv8 模型文件。

### 2. CUDA 错误

如果遇到 CUDA 相关错误，可以：

- 检查 CUDA 版本是否与 PyTorch 版本匹配
- 在 config.yaml 中设置 `device: "cpu"` 使用 CPU 模式

### 3. 视频流无法打开

检查：

- 视频流 URL 是否正确
- 网络连接是否正常
- OpenCV 是否支持该流格式

### 4. 内存不足

- 降低 `max_fps` 和增加 `frame_skip`
- 使用较小的 YOLO 模型（yolov8n/s）

## 与 SpringBoot 集成

在 SpringBoot 中调用 AI 服务：

```java
@Service
public class AiDetectionService {

    @Value("${ai.service.url}")
    private String aiServiceUrl; // http://localhost:5000

    public DetectionResult detectImage(MultipartFile file) {
        RestTemplate restTemplate = new RestTemplate();

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", file.getResource());
        body.add("conf_threshold", 0.25);
        body.add("return_image", true);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> request =
            new HttpEntity<>(body, headers);

        ResponseEntity<DetectionResult> response = restTemplate.postForEntity(
            aiServiceUrl + "/api/detect",
            request,
            DetectionResult.class
        );

        return response.getBody();
    }
}
```

## 开发调试

启用开发模式（自动重载）：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 5000
```

## 许可证

本项目基于 MIT 许可证。

## 联系方式

如有问题或建议，请联系项目维护者。
