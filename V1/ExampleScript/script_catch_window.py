import pygetwindow as gw
import pyautogui
from PIL import ImageGrab
import time


def capture_active_window(save_path="focused_window.png"):
    # 获取当前聚焦窗口
    active_window = gw.getActiveWindow()
    if active_window is None:
        print("未找到当前聚焦窗口")
        return

    # 窗口坐标和尺寸
    left = active_window.left
    top = active_window.top
    right = left + active_window.width
    bottom = top + active_window.height

    # 截图
    img = ImageGrab.grab(bbox=(left, top, right, bottom))
    img.save(save_path)
    print(f"截图保存到 {save_path}")


# 示例调用
time.sleep(2)  # 给你2秒切到目标窗口
capture_active_window()
