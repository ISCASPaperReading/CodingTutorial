import torch


# 注意loss_func不一定通过传参形式给到trainer, 可以直接import
# device 是CPU或CUDA, 或者特定编号的GPU
def trainer(x, y, model, optimizer, loss_func, device):
    # 将模型参数设为训练模式
    model.train()
    # 将数据存入对应设备中
    x = x.to(device)
    y = y.to(device)
    # 梯度清零
    optimizer.zero_grad()
    # 前向传播
    y_hat = model(x)
    # 计算loss
    loss = loss_func(y_hat, y)
    # 反向传播获取梯度
    loss.backward()
    # 更新参数
    optimizer.step()

    # 计算准确率
    #predictions = torch.argmax(y_hat, dim=1)  # 获取模型的预测结果
    correct_predictions = (y_hat == y).sum().item()  # 统计正确的预测数量
    return loss.item() / y.shape[0], correct_predictions


def validate(val_loader, model, loss_func, device):
    # 将模型参数设为评估模式
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算
        for x,y,x_mark,y_mark in val_loader:
            x = x.to(device)
            y = y.to(device)
            y = y[:,-96:,:]

            # 前向传播
            y_hat = model(x)
            # 计算loss
            loss = loss_func(y_hat, y)

            total_loss += loss.item() / y.size(0)

            # 计算准确率
            correct_predictions += (y_hat == y).sum().item()
            total_samples += y.size(0)

    # 计算平均损失和准确率
    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy
