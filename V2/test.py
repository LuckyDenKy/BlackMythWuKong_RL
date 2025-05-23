import time

import pygetwindow as gw
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import pyautogui

keyboard = KeyboardController()
mouse = MouseController()
# game_window = gw.getWindowsWithTitle("b1")[0]
# if not game_window.isActive:
#     game_window.activate()
#     time.sleep(5)
time.sleep(5)
print("开始移动")
for _ in range(10):
    # mouse.move(200, 0)

    pyautogui.moveRel(10,0,duration=0.1)
    time.sleep(0.5)
