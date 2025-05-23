from torch import nn
from torchvision import models
import torch.nn.functional as F

def print_parameters(model):
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        total_params += param.numel()
    print(f"Total parameters: {total_params}, trainable parameters: {trainable_params}")

class ED_v1(nn.Module):
    def __init__(self,num_class=2):
        super().__init__()
        self.base_model = models.resnet18()
        self.classifier = nn.Linear(self.base_model.fc.out_features,num_class)

    def forward(self,x):
        assert x.shape[1] == 3
        hidden_states = F.relu(self.base_model(x))
        prob = F.softmax(self.classifier(hidden_states),dim=1)
        return prob

if __name__ == '__main__':
    device = "cuda"
    model = ED_v1().to(device)
    print(model)
    print_parameters(model)
    input("回车结束...")



