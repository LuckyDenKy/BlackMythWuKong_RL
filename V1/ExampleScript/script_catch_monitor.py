import mss
from PIL import Image
import torchvision.transforms as transforms
import torch

# 定义截图区域（显示器1，主屏幕）
with mss.mss() as sct:
    monitor = sct.monitors[1]  # 1是主屏，2是扩展屏

    # 获取主屏截图
    screenshot = sct.grab(monitor)

    # 将 raw 像素数据转换为 PIL 图像（RGB）
    img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)

    # 显示图像（调试用）
    img.show()

    # 转换为Tensor（C, H, W），值范围在 [0, 1]
    transform = transforms.ToTensor()
    tensor_image = transform(img)

    print("图像 tensor 的形状:", tensor_image.shape)
    print("类型:", type(tensor_image))
