import os.path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,CyclicLR

transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 先缩放稍大一点
        transforms.RandomCrop((128, 128)),  # 随机裁剪，增加鲁棒性
        transforms.RandomHorizontalFlip(),  # 左右翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # 标准化（ImageNet的均值和方差）
                             [0.229, 0.224, 0.225])
    ])

def build_dataset(dataset_folder="dataset"):
    dataset = datasets.ImageFolder(dataset_folder, transform=transform)
    return dataset

def build_data_train_loader(dataset,batch_size=1,shuffle=True):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader

# 自定义CNN
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*32*32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self,x):
        return self.model(x)


class ImprovedSmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # (B, 32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # (B, 32, 64, 64)

            nn.Conv2d(32, 64, 3, padding=1),  # (B, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # (B, 64, 32, 32)

            nn.Conv2d(64, 128, 3, padding=1), # (B, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # (B, 128, 16, 16)

            nn.AdaptiveAvgPool2d((4, 4)),     # (B, 128, 4, 4)
            nn.Flatten(),                     # (B, 2048)
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)

class BiggerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),   # 通道从16->64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), # 通道32->128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), # 新增一层，256通道
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(256 * 16 * 16, 512),  # 假设输入图像128x128，经过3次池化后大小是16x16
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 2)  # 2分类输出
        )

    def forward(self, x):
        return self.model(x)

class PretrainedModelWrapper(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=2, freeze_backbone=True):
        super().__init__()
        self.model = self._load_and_prepare_model(model_name, num_classes, freeze_backbone)

    def _load_and_prepare_model(self, model_name, num_classes, freeze_backbone):
        model = getattr(models, model_name)(pretrained=True)

        if 'resnet' in model_name:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif 'mobilenet' in model_name:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif 'efficientnet' in model_name:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported yet.")

        if freeze_backbone:
            for name, param in model.named_parameters():
                if not any(k in name for k in ['fc', 'classifier']):
                    param.requires_grad = False

        return model

    def forward(self, x):
        return self.model(x)


def build_model(device,model_name):
    if model_name == 'SmallCNN':
        model = SmallCNN().to(device)
    elif model_name == 'ImprovedSmallCNN':
        model = ImprovedSmallCNN().to(device)
    elif model_name == 'BiggerCNN':
        model = BiggerCNN().to(device)
    elif model_name in ['resnet18','resnet50']:
        model = PretrainedModelWrapper(model_name).to(device)
    else:
        return None
    return model

def train(model_name,epochs=10):
    data_set = build_dataset('dataset')
    data_loader = build_data_train_loader(data_set,batch_size=2,shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device,model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-1)

    # 学习率调度器：监控准确率（也可换成 val_loss）
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    # 定义CyclicLR调度器
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-5,  # 学习率循环的下界（最低值）
        max_lr=1e-1,  # 学习率循环的上界（最高值）
        step_size_up=78,  # 学习率从base_lr升到max_lr所用的训练batch数
        mode='triangular2',  # 学习率曲线形状，'triangular'、'triangular2' 或 'exp_range'
        cycle_momentum=False  # Adam不用调整momentum，设False
    )

    model.train()
    max_acc = 0
    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, LR: {optimizer.param_groups[0]['lr']}")
        if accuracy > max_acc:
            ckpt_name = f'{accuracy:.2f}-{model_name}-weights.pth'
            torch.save(model.state_dict(), os.path.join('enemy_detect_ckpt',ckpt_name))
            max_acc = accuracy
            print(f"模型已保存到: {ckpt_name}")

        # # 根据准确率调整学习率
        # scheduler.step(accuracy)

def detect_enemy(image_pil,device,model):
    image = transform(image_pil).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    return pred == 1  # True 表示有小怪


if __name__ == '__main__':
    torch.manual_seed(2025)
    train('resnet50',50)