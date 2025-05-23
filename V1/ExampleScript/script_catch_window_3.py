import pygetwindow as gw
import pyautogui
from PIL import Image
window_title = "b1"

try:
    window = gw.getWindowsWithTitle(window_title)[0]
except IndexError:
    print(f"未找到标题为'{window_title}'的窗口")
    exit()

x,y = window.topleft
x,y = x+11,y+45
width,height = window.size
width,height = width-22,height-57

screenshot = pyautogui.screenshot(region=(x,y,width,height))
screenshot.save("screenshot3.png") # 1280 719

# 捕获状态（血条，蓝条，气力）
x_state, y_state = 138, 650
width_state, height_state = 237, 33  # 221, 12
# 裁剪出血条区域图像
state_bar_region = screenshot.crop(
    (x_state, y_state, x_state + width_state, y_state + height_state)
)
# 保存或显示裁剪结果
state_bar_region.save("state_bar.png")

# 血条状态
x_blood, y_blood = 138,653
width_blood, height_blood = 221, 11
blood_bar_region = screenshot.crop(
    (x_blood,y_blood,x_blood+width_blood,y_blood+height_blood)
)
blood_bar_region.save("blood_bar.png")
# blood_bar_region.show()

# 蓝条
x_magic, y_magic = 139,668
width_magic, height_magic = 232, 6
magic_bar_region = screenshot.crop(
    (x_magic,y_magic,x_magic+width_magic,y_magic+height_magic)
)
magic_bar_region.save("magic_bar.png")
# magic_bar_region.show()

# 气力
x_strength, y_strength = 139,677
width_strength, height_strength = 235, 5
strength_bar_region = screenshot.crop(
    (x_strength,y_strength,x_strength+width_strength,y_strength+height_strength)
)
strength_bar_region.save("strength_bar.png")
# strength_bar_region.show()

print("已保存截图")