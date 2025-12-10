import subprocess
import re

# 你指定的待检查编码器列表（FourCC/别名 → 容器格式）
codecs_to_try = [
    ('avc1', 'mp4'),   # H.264 for MP4 container
    ('H264', 'mp4'),   # H.264 alternative
    ('X264', 'mp4'),   # x264 encoder
    ('XVID', 'avi'),   # Xvid (fallback, change extension)
    ('mp4v', 'mp4'),   # MPEG-4 Part 2 (last resort)
]

# FFmpeg编码器名称映射（解决命名不一致问题）
codec_mapping = {
    'avc1': ['h264', 'libx264'],
    'H264': ['h264', 'libx264'],
    'X264': ['libx264'],
    'XVID': ['libxvid'],
    'mp4v': ['mpeg4']
}

def get_available_encoders():
    """调用FFmpeg获取所有可用编码器名称"""
    try:
        # 执行ffmpeg -encoders并捕获输出
        result = subprocess.check_output(
            ['ffmpeg', '-encoders'],
            stderr=subprocess.STDOUT,  # 重定向stderr到stdout
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        # 提取所有编码器名称（匹配"E..."开头的行，如：E libx264  H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10）
        encoder_lines = re.findall(r'E\s+(\w+)\s+', result)
        return set(encoder_lines)  # 去重并返回集合
    except FileNotFoundError:
        print("错误：未找到FFmpeg，请确认已安装并配置环境变量！")
        return set()
    except Exception as e:
        print(f"获取编码器列表失败：{e}")
        return set()

def check_codecs():
    """检查指定编码器是否可用"""
    available_encoders = get_available_encoders()
    if not available_encoders:
        return

    print("=== 编码器检查结果 ===")
    for codec_name, container in codecs_to_try:
        # 获取该编码器在FFmpeg中的实际名称
        ffmpeg_names = codec_mapping.get(codec_name, [codec_name.lower()])
        # 检查是否存在匹配的编码器
        exists = any(name in available_encoders for name in ffmpeg_names)
        
        status = "✅ 存在" if exists else "❌ 不存在"
        print(f"{codec_name}（容器：{container}）: {status}")
        if exists:
            matched = [name for name in ffmpeg_names if name in available_encoders]
            print(f"  → 匹配到FFmpeg编码器：{', '.join(matched)}")

if __name__ == "__main__":
    check_codecs()