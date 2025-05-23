import ctypes
import win32gui
import win32ui
import win32con
import win32api
import numpy as np
import cv2

def capture_foreground_window_opencv(save_path="screenshot.png"):
    hwnd = win32gui.GetForegroundWindow()
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # ⛳ 关键改动：使用 ctypes 调用 PrintWindow
    PW_RENDERFULLCONTENT = 0x00000002
    result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), PW_RENDERFULLCONTENT)

    # 获取截图图像
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype='uint8')
    img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)  # BGRA

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(save_path, img_rgb)
        print(f"截图成功：{save_path}")
        return img_rgb
    else:
        print("截图失败（PrintWindow 返回 0）")
        return None

# 调用测试
capture_foreground_window_opencv()
