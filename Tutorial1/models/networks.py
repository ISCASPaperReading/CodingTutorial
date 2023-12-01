import torch
import torch.nn as nn
from models.layers import create_layers

# 定义网络
class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = create_layers(784,256,10)
    def forward(self,x):
        B,C,W,H = x.shape
        x = x.view(B,W*H)
        return self.net(x)