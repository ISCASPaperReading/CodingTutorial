"""
import 应该分为三部分
1. python 内置模块
2. 来自第三方库的模块
3. 自定义模块
"""

from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.networks import TestNet
from datasets.datasets import get_mnist_dataset
from trainers import trainer, validate


if __name__ == "__main__":
    # 初始化
    model = TestNet().cuda()
    optimizer = Adam(model.parameters(),lr=1e-3)
    loss_func = F.cross_entropy

    train_dataset, test_dataset = get_mnist_dataset()
    # 定义 DataLoader 来加载数据
    batch_size = 512
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    device = "cuda"
    total_ep = 100
    for ep in range(total_ep):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in train_loader:
            loss, correct_predictions = trainer(batch, model, optimizer, loss_func, device)
            total_loss += loss
            total_correct += correct_predictions
            total_samples += batch[1].shape[0]
        average_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        print(f"Epoch: {ep} Training Loss: {average_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
            
        if (ep+1) % 5 == 0:
            validate(test_loader, model, loss_func, device)