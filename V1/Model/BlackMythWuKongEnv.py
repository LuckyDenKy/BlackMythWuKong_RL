import time
import torch
import pygetwindow as gw
import pyautogui
import numpy as np
from torchvision import transforms
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import threading

try:
    from EnemyDetection import ED_v1
except:
    from Model.EnemyDetection import ED_v1

class BlackMythWuKongEnv:
    '''
        游戏画面需要窗口化设置，分辨率调到最低
    '''
    def __init__(self,device,ed_lr=1e-5):
        # 获取游戏窗口句柄
        try:
            self.game_window = gw.getWindowsWithTitle("b1")[0]
            # self._activate_window()
            # self._keep_active = True  # 控制线程运行的标志
            # # 启动守护线程：定期检查窗口是否激活
            # self._watchdog_thread = threading.Thread(
            #     target=self._watchdog_keep_active,
            #     daemon=True  # 主程序退出时自动结束线程
            # )
            # self._watchdog_thread.start()

        except IndexError:
            print(f"未找到游戏窗口")
            exit()

        # 窗口捕获信息
        self.X, self.Y = self.game_window.topleft
        self.X, self.Y = self.X + 11, self.Y + 45
        self.Width, self.Height = self.game_window.size
        self.Width, self.Height = self.Width - 22, self.Height - 57

        # 血条状态窗口信息
        self.x_blood, self.y_blood = 138, 653
        self.width_blood, self.height_blood = 221, 11

        # 蓝条状态窗口信息
        self.x_magic, self.y_magic = 139, 668
        self.width_magic, self.height_magic = 232, 6

        # 气力状态窗口信息
        self.x_strength, self.y_strength = 139, 677
        self.width_strength, self.height_strength = 235, 5

        # 血条颜色列表（近白灰）
        self.blood_colors = [(218, 217, 214),(202, 200, 199),(186, 183, 178)]
        # 蓝条颜色列表（蓝）
        self.magic_colors = [(65, 113, 171),(59, 103, 158),(43, 72, 109)]
        # 气力颜色列表（黄色）
        self.strength_colors = [(191, 156, 101),(184, 150, 97),(175, 143, 93)]

        # 颜色容忍度
        self.threshold = 18

        # 怪物检测网络，返回2个值分别标识有怪物和无怪物的概率
        self.device = device
        self.enemy_detect = ED_v1(num_class=2).to(device)
        self.ed_optimizer = torch.optim.Adam(self.enemy_detect.parameters(),lr=ed_lr)
        self.transform = transforms.Compose([
            transforms.Resize((224,126)),  # (224,126) (512,288)
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # 标准化（ImageNet的均值和方差）
                                 [0.229, 0.224, 0.225])
        ])

        # 动作信息
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.action_dict = {
            'W': lambda: (self.keyboard.press('w'),time.sleep(2),self.keyboard.release('w')),
            # 'A': lambda: (self.keyboard.press('a'),time.sleep(2),self.keyboard.release('a')),
            # 'S': lambda: (self.keyboard.press('s'),time.sleep(2),self.keyboard.release('s')),
            # 'D': lambda: (self.keyboard.press('d'),time.sleep(2),self.keyboard.release('d')),
            'Space': lambda: self.keyboard.press(Key.space) or self.keyboard.release(Key.space),
            'mouse_left': lambda: (self.mouse.click(Button.left),
                                   # time.sleep(0.5)
                                   )
        }
        self.action_list = list(self.action_dict.keys())
        self.action_dim = len(self.action_list)
        self.press_mouse_middle = lambda: (self.mouse.click(Button.middle),time.sleep(0.2))

        # 状态信息
        self.state_dim = 4  # 血条+蓝条+有怪物概率+无怪物概率
        self.current_state = None
        self.loss_blood = False

    def step(self,action):
        # 执行动作action，返回下一状态，计算奖励，是否完成
        # 1. 执行动作
        self._activate_window()
        action = self.action_list[action]
        print(f"执行动作:{action}")
        self.press_mouse_middle()  # 每次都按锁定一下敌人
        if action in self.action_list:
            self.action_dict[action]()
        else:
            self.action_dict['W']()
        # 2. 获取下一状态
        old_state = self.current_state
        next_state = self.get_state()
        # 3. 计算奖励
        delta_blood = next_state[0].item()-old_state[0].item()
        delta_magic = next_state[1].item()-old_state[1].item()
        reward = delta_blood + delta_magic  # 血状态奖励
        if self.loss_blood and action == 'mouse_left':
            reward += 5.0
        elif self.loss_blood and action == 'Space':
            reward += 0.5
        else:
            reward += 0.1

        # 4. 是否完成
        done = True if next_state[0].item() < 0.01 else False

        # 5. 其它信息
        info = {}
        return next_state, reward, done, info

    def update(self,transition_dict):
        states = torch.stack(transition_dict['states'], dim=0).to(self.device)
        next_states = torch.stack(transition_dict['next_states'], dim=0).to(self.device)

        blood_states = states[:,0]
        blood_next_states = next_states[:,0]
        delta_blood = blood_next_states - blood_states

        has_enemy = next_states[:,-2]
        no_enemy = next_states[:,-1]

        loss = (delta_blood*(has_enemy / (no_enemy + 1e-5))).mean()
        # print(loss)
        self.ed_optimizer.zero_grad()
        loss.backward()
        self.ed_optimizer.step()

    def get_state(self):
        # 获取一帧
        screenshot = pyautogui.screenshot(region=(self.X, self.Y, self.Width, self.Height))
        # 获取血条，蓝条和气力状态值(这里暂时实现血条和蓝条)
        blood,magic,strength = self.get_blood_magic_strength(screenshot)
        if strength is None:
            state1 = torch.tensor([blood,magic],device=self.device,dtype=torch.float)
        else:
            state1 = torch.tensor([blood,magic,strength],device=self.device,dtype=torch.float)
        # 获取怪物信息
        enemy_probs = self.get_enemy(screenshot)
        state = torch.cat((state1,enemy_probs))
        self.current_state = state
        return state  # tensor类型

    def get_enemy(self,screenshot):
        figure_tensor = self.transform(screenshot).to(self.device)
        probs = self.enemy_detect(figure_tensor.unsqueeze(0)).squeeze(0)
        return probs

    def get_blood_magic_strength(self,screenshot):
        # 血条状态
        blood_bar_region = screenshot.crop(
            (self.x_blood, self.y_blood, self.x_blood + self.width_blood, self.y_blood + self.height_blood)
        )

        # 蓝条状态
        magic_bar_region = screenshot.crop(
            (self.x_magic, self.y_magic, self.x_magic + self.width_magic, self.y_magic + self.height_magic)
        )

        # # 气力
        # strength_bar_region = screenshot.crop(
        #     (self.x_strength, self.y_strength, self.x_strength + self.width_strength, self.y_strength + self.height_strength)
        # )

        blood_ratio = self.get_bar_ratio_multi_color(blood_bar_region, self.blood_colors, threshold=self.threshold)
        magic_ratio = self.get_bar_ratio_multi_color(magic_bar_region, self.magic_colors, threshold=self.threshold)
        # strength_ratio = self.get_bar_ratio_multi_color(strength_bar_region, self.strength_colors, threshold=self.threshold)

        return blood_ratio,magic_ratio,None

    def get_bar_ratio_multi_color(self,pil_img, target_colors, threshold=30):
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

    def _activate_window(self):
        """强制激活目标窗口"""
        try:
            if not self.game_window.isActive:
                self.game_window.activate()
                time.sleep(0.1)  # 避免快速切换
        except Exception as e:
            print(f"激活窗口失败: {e}")

    def _watchdog_keep_active(self):
        """守护线程：每0.5秒检查一次窗口状态"""
        while self._keep_active:
            if not self.game_window.isActive:
                self._activate_window()
            time.sleep(0.5)  # 检查间隔

    def print_action_idx(self):
        for i in range(self.action_dim):
            print(f'{self.action_list[i]}--{i}')

if __name__ == '__main__':
    device = "cuda"
    env = BlackMythWuKongEnv(device=device)
    state = env.get_state()

    # print(f"状态信息：{state}")wsw