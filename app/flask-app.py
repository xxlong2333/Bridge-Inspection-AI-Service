import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from torch import nn
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import time
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# 配置
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB文件大小限制

# 检查CUDA设备是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# AODNet模型定义
class dehaze_net(nn.Module):
    def __init__(self):
        super(dehaze_net, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        source = []
        source.append(x)
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))
        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))
        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))
        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image


# 加载模型
def load_models():
    # 临时解决 PyTorch 2.6+ 的 weights_only 问题
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)

    try:
        # 加载YOLO模型
        # 获取当前文件所在目录的父目录路径
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, 'best.pt')
        yolo_model = YOLO(model_path).to(device)

        # # 加载AODNet去雾模型
        # aodnet = dehaze_net().to(device)
        # aodnet.load_state_dict(torch.load(r'D:\WorkPlace\pythonProgram\Aodnet\bestmodel\best_model.pth'))
        # aodnet.eval()
# , aodnet
        return yolo_model
    finally:
        # 恢复原始的 torch.load 方法
        torch.load = original_load


# 全局变量用于存储处理进度
progress_data = {
    'current': 0,
    'total': 1,
    'percentage': 0
}

# 加载模型
model, dehaze_model = load_models()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def update_progress(current, total):
    percentage = int((current / total) * 100)
    progress_data.update({
        'current': current,
        'total': total,
        'percentage': percentage
    })
    socketio.emit('progress_update', progress_data)


def preprocess_image(image):
    """将OpenCV图像转换为模型输入张量"""
    # 转换BGR到RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 归一化到0-1并转换为张量
    image = torch.from_numpy(image).float() / 255.0
    # 调整维度顺序为CxHxW
    image = image.permute(2, 0, 1)
    # 添加batch维度
    image = image.unsqueeze(0)
    return image


def postprocess_image(tensor):
    """将模型输出张量转换回OpenCV图像"""
    # 移除batch维度
    tensor = tensor.squeeze(0)
    # 调整维度顺序为HxWxC
    tensor = tensor.permute(1, 2, 0)
    # 转换回numpy数组
    image = tensor.cpu().numpy()
    # 缩放回0-255范围
    image = (image * 255).astype('uint8')
    # 转换RGB到BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def dehaze_image(image):
    """使用AODNet对图像去雾"""
    with torch.no_grad():
        # 预处理
        input_tensor = preprocess_image(image).to(device)
        # 去雾处理
        output_tensor = dehaze_model(input_tensor)
        # 后处理
        dehazed_image = postprocess_image(output_tensor)
    return dehazed_image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='未选择文件')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='未选择文件')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                # 保存上传文件
                file.save(upload_path)

                start_time = time.time()

                # 根据文件类型处理
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    result_path = process_image(upload_path, filename)
                else:
                    result_path = process_video(upload_path, filename)

                process_time = time.time() - start_time

                if not result_path:
                    raise RuntimeError("未能生成结果文件")

                return render_template(
                    'index.html',
                    input_file=url_for('static', filename=f'uploads/{filename}'),
                    result_file=url_for('static', filename=f'results/{os.path.basename(result_path)}'),
                    timestamp=int(time.time()),  # 添加时间戳防止缓存
                    process_time=f'{process_time:.2f}秒',
                    error=None
                )

            except Exception as e:
                # 清理可能存在的临时文件
                if os.path.exists(upload_path):
                    os.remove(upload_path)
                return render_template('index.html', error=f'处理失败: {str(e)}')

    return render_template('index.html', error=None)


def process_image(image_path, filename):
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("无法读取图像文件")

        # 去雾处理
        image = dehaze_image(image)

        # 目标检测
        results = model.predict(image, device=device)

        # 保存结果
        result_path = os.path.join(app.config['RESULT_FOLDER'], f'result_{filename}')
        for r in results:
            im_array = r.plot()
            cv2.imwrite(result_path, cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR))

        return result_path
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return None


# 全局进度变量
progress_data = {
    'current': 0,
    'total': 1,
    'percentage': 0,
    'is_processing': False
}

# 添加新的路由获取进度
@app.route('/get_progress')
def get_progress():
    return jsonify(progress_data)

def update_progress(current, total):
    percentage = int((current / total) * 100)
    progress_data.update({
        'current': current,
        'total': total,
        'percentage': percentage,
        'is_processing': True
    })
    socketio.emit('progress_update', progress_data)

# 在处理结束时重置状态
def reset_progress():
    progress_data.update({
        'current': 0,
        'total': 1,
        'percentage': 0,
        'is_processing': False
    })
    socketio.emit('progress_update', progress_data)

def process_video(video_path, filename):
    try:
        # 开始处理前重置状态
        reset_progress()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        update_progress(0, total_frames)



        # 获取视频参数
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 更新进度
        update_progress(0, total_frames)

        # 使用H.264编码
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        result_filename = f'result_{os.path.splitext(filename)[0]}.mp4'
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        # 创建视频写入器
        out = cv2.VideoWriter(result_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise RuntimeError("无法创建输出视频文件")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 去雾处理
            frame = dehaze_image(frame)

            # 执行检测
            results = model.predict(frame, device=device)

            # 绘制结果
            for r in results:
                frame = r.plot()

            # 写入帧
            out.write(frame)
            frame_count += 1

            # 更新进度
            update_progress(frame_count, total_frames)

            # 每处理50帧打印进度
            if frame_count % 50 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧 ({progress_data['percentage']}%)")

        cap.release()
        out.release()

        # 验证输出文件
        if not os.path.exists(result_path):
            raise FileNotFoundError("输出视频未生成")

        if os.path.getsize(result_path) < 1024:  # 至少1KB
            raise ValueError("输出视频文件过小")

        print(f"视频处理完成，保存到: {result_path}")
        reset_progress()

        return result_path

    except Exception as e:
        # 清理不完整的输出文件
        if 'result_path' in locals() and os.path.exists(result_path):
            os.remove(result_path)
        print(f"视频处理失败: {str(e)}")
        raise


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    socketio.run(app,
                host='0.0.0.0',
                port=5000,
                debug=True,
                allow_unsafe_werkzeug=True)  # 添加这个参数