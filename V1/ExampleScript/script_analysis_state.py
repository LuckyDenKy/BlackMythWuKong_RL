import time
import pygetwindow as gw
import pyautogui
from PIL import Image
import numpy as np

window_title = "b1"

try:
    window = gw.getWindowsWithTitle(window_title)[0]
except IndexError:
    print(f"未找到标题为'{window_title}'的窗口")
    exit()

# x,y = window.topleft
# x,y = x+11,y+45
# width,height = window.size
# width,height = width-22,height-57
#
# screenshot = pyautogui.screenshot(region=(x,y,width,height))
#
# # 定义三种状态
# blood_bar_region,magic_bar_region,strength_bar_region = None,None,None
# # 血条状态
# x_blood, y_blood = 138,653
# width_blood, height_blood = 221, 11
# blood_bar_region = screenshot.crop(
#     (x_blood,y_blood,x_blood+width_blood,y_blood+height_blood)
# )
#
# # 蓝条
# x_magic, y_magic = 139,668
# width_magic, height_magic = 232, 6
# magic_bar_region = screenshot.crop(
#     (x_magic,y_magic,x_magic+width_magic,y_magic+height_magic)
# )
#
# # 气力
# x_strength, y_strength = 139,677
# width_strength, height_strength = 235, 5
# strength_bar_region = screenshot.crop(
#     (x_strength,y_strength,x_strength+width_strength,y_strength+height_strength)
# )
#
# # 血条颜色列表（近白灰）
# blood_colors = [
#     (218, 217, 214),
#     (202, 200, 199),
#     (186, 183, 178)
# ]
# # 蓝条颜色列表（蓝）
# magic_colors = [
#     (65, 113, 171),
#     (59, 103, 158),
#     (43, 72, 109)
# ]
# # 气力颜色列表（黄色）
# strength_colors = [
#     (191, 156, 101),
#     (184, 150, 97),
#     (175, 143, 93)
# ]

# # 计算三个条的比例
# threshold = [15,16,17,18,19,20]
# for t in threshold:
#     blood_ratio = get_bar_ratio_multi_color(blood_bar_region, blood_colors, threshold=t)
#     magic_ratio = get_bar_ratio_multi_color(magic_bar_region, magic_colors, threshold=t)
#     strength_ratio = get_bar_ratio_multi_color(strength_bar_region, strength_colors, threshold=t)
#     print("threshold:",t)
#     print("血量比例:", blood_ratio)
#     print("蓝量比例:", magic_ratio)
#     print("气力比例:", strength_ratio)
#     print()

# 初始状态
# threshold: 15
# 血量比例: 0.9811
# 蓝量比例: 0.9792
# 气力比例: 0.9915
#
# threshold: 16
# 血量比例: 0.9885
# 蓝量比例: 0.9892
# 气力比例: 0.9915
#
# threshold: 17
# 血量比例: 0.9889
# 蓝量比例: 0.9914
# 气力比例: 0.9915
#
# threshold: 18
# 血量比例: 0.9897
# 蓝量比例: 0.9914
# 气力比例: 0.9915
#
# threshold: 19
# 血量比例: 0.9901
# 蓝量比例: 0.9921
# 气力比例: 0.9915
#
# threshold: 20
# 血量比例: 0.9905
# 蓝量比例: 0.9928
# 气力比例: 0.9923

def get_bar_ratio_multi_color(pil_img, target_colors, threshold=30):
    """
    提取状态条的填充比例（支持多个目标颜色）。
    参数:
        pil_img: PIL.Image.Image 类型，RGB 模式图像
        target_colors: List[Tuple(R,G,B)]，状态条可能的多个主色值
        threshold: int，颜色容差阈值（颜色距离）

    返回:
        ratio: float，填充比例（0~1）
    """
    img = np.array(pil_img)
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=bool)

    for color in target_colors:
        dist = np.linalg.norm(img - np.array(color), axis=2)
        mask |= (dist < threshold)

    filled_pixels = np.sum(mask)
    total_pixels = h * w
    ratio = filled_pixels / total_pixels
    return round(ratio, 2)

def get_state(count=10):
    # 窗口捕获信息
    x, y = window.topleft
    x, y = x + 11, y + 45
    width, height = window.size
    width, height = width - 22, height - 57
    # 血条状态捕获信息
    x_blood, y_blood = 138, 653
    width_blood, height_blood = 221, 11
    # 蓝条状态捕获信息
    x_magic, y_magic = 139, 668
    width_magic, height_magic = 232, 6
    # 气力状态捕获信息
    x_strength, y_strength = 139, 677
    width_strength, height_strength = 235, 5
    # 三种状态比例条
    blood_bar_region, magic_bar_region, strength_bar_region = None, None, None
    # 血条颜色列表（近白灰）
    blood_colors = [
        (218, 217, 214),
        (202, 200, 199),
        (186, 183, 178)
    ]
    # 蓝条颜色列表（蓝）
    magic_colors = [
        (65, 113, 171),
        (59, 103, 158),
        (43, 72, 109)
    ]
    # 气力颜色列表（黄色）
    strength_colors = [
        (191, 156, 101),
        (184, 150, 97),
        (175, 143, 93)
    ]

    for i in range(count):
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        # 血条状态
        blood_bar_region = screenshot.crop(
            (x_blood, y_blood, x_blood + width_blood, y_blood + height_blood)
        )

        # 蓝条
        magic_bar_region = screenshot.crop(
            (x_magic, y_magic, x_magic + width_magic, y_magic + height_magic)
        )

        # 气力
        strength_bar_region = screenshot.crop(
            (x_strength, y_strength, x_strength + width_strength, y_strength + height_strength)
        )

        blood_ratio = get_bar_ratio_multi_color(blood_bar_region, blood_colors, threshold=18)  # 选用threshold=18
        magic_ratio = get_bar_ratio_multi_color(magic_bar_region, magic_colors, threshold=18)
        strength_ratio = get_bar_ratio_multi_color(strength_bar_region, strength_colors, threshold=18)
        print(f"血量比例: {blood_ratio},蓝量比例: {magic_ratio},气力比例: {strength_ratio},")
        time.sleep(2)

get_state()