"""
图像处理工具函数
"""
import base64
from typing import Optional
import numpy as np
import cv2
from io import BytesIO
from PIL import Image


def encode_image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """
    将OpenCV图像编码为Base64字符串

    Args:
        image: OpenCV图像 (BGR格式)
        format: 图像格式 (JPEG, PNG)

    Returns:
        Base64编码字符串
    """
    # BGR转RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 转换为PIL Image
    pil_img = Image.fromarray(image)

    # 编码为字节流
    buffered = BytesIO()
    pil_img.save(buffered, format=format)

    # Base64编码
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_base64


def decode_base64_to_image(base64_str: str) -> Optional[np.ndarray]:
    """
    将Base64字符串解码为OpenCV图像

    Args:
        base64_str: Base64编码字符串

    Returns:
        OpenCV图像 (BGR格式) 或 None
    """
    try:
        # 去除data URL前缀
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        # Base64解码
        img_bytes = base64.b64decode(base64_str)

        # 转换为numpy数组
        nparr = np.frombuffer(img_bytes, np.uint8)

        # 解码为OpenCV图像
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return image

    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    调整图像大小

    Args:
        image: 输入图像
        width: 目标宽度
        height: 目标高度
        keep_aspect_ratio: 是否保持宽高比

    Returns:
        调整大小后的图像
    """
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if keep_aspect_ratio:
        if width is not None:
            ratio = width / w
            height = int(h * ratio)
        elif height is not None:
            ratio = height / h
            width = int(w * ratio)
    else:
        width = width or w
        height = height or h

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized
