from pynput.keyboard import Controller, Key
import time

keyboard = Controller()

def wait_for_focus():
    """等待用户手动聚焦到目标窗口"""
    print("请手动点击文本文件窗口（5秒等待时间）")
    time.sleep(5)  # 留出时间切换窗口

def simulate_typing(text):
    """模拟键盘输入（支持空格、回车等特殊键）"""
    for char in text:
        if char == ' ':
            keyboard.press(Key.space)
            keyboard.release(Key.space)
        elif char == '\n':
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)
        else:
            keyboard.press(char)
            keyboard.release(char)
        time.sleep(0.1)  # 每个字符间隔0.1秒
    print("输入完成！")

def main():
    # 直接在代码中定义要输入的内容（示例）
    content = "郭超奇\nSB\nTesting!\n杨硕\n还在火焰山\nTesting!\n"
    
    # 提示用户准备
    print(f"即将输入的内容：\n{content}\n")
    wait_for_focus()
    
    # 开始模拟输入
    simulate_typing(content)

if __name__ == "__main__":
    main()