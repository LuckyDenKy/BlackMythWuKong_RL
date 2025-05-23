import time
# import mss
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models
import pygetwindow as gw
import pyautogui

window_title = "b1"
try:
    window = gw.getWindowsWithTitle(window_title)[0]
    x, y = window.topleft
    width, height = window.size
except IndexError:
    print(f"未找到标题为'{window_title}'的窗口")
    exit()

# 设置设备和特征提取网络
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # 去掉最后一层全连接分类头
resnet = resnet.to(device)
resnet.eval()

def catch_screen_to_tensor(index=None):
    img = pyautogui.screenshot(region=(x+11,y+45,width-22,height-57))
    # img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
    if index is not None:
        img.save(f'catch_imgs/{index}.png')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(  # ImageNet 均值和方差
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)  # 加 batch 维度 (1, 3, 224, 224)

    # 4. 提取特征
    with torch.no_grad():
        features = resnet(input_tensor)  # 输出为 (1, 2048)

    # print("特征 shape:", features.shape)
    # print("特征向量:", features.cpu().numpy())
    return features

if __name__ == '__main__':
    print("准备捕获，等待5s")
    time.sleep(5)
    print('开始捕获10帧')
    for i in range(10):
        catch_screen_to_tensor(i)