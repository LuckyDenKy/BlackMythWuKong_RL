import time
import pygetwindow as gw
import pyautogui
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

# screenshot = pyautogui.screenshot(region=(x,y,width,height))
# screenshot.save("screenshot3.png") # 1280 719

print("开始捕获图像...")
for i in range(10000):
    time.sleep(2)
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    screenshot.save(f"catch_imgs/{i}.png")
    print(f"已保存{i}.png")