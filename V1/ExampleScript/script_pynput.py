from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

# 创建控制器
keyboard = KeyboardController()
mouse = MouseController()

# 定义动作列表（每个元素是一个函数或 lambda 表达式）
actions = [
    lambda: keyboard.press('w') or keyboard.release('w'),
    lambda: keyboard.press('a') or keyboard.release('a'),
    lambda: keyboard.press('s') or keyboard.release('s'),
    lambda: keyboard.press('d') or keyboard.release('d'),
    lambda: keyboard.press(Key.space) or keyboard.release(Key.space),
    lambda: mouse.click(Button.left)
]

# 示例：依次执行所有动作（每个动作间暂停 0.5 秒）
import time
for action in actions:
    action()
    time.sleep(0.5)
