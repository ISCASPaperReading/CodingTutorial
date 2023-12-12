from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.net import TestNet
from data.Datasets import MyDataset, get_dataset
from train import trainer, validate

if __name__ == '__main__':
    root_path = './data/'
    data_path = 'weather.csv'
    seq_len, label_len, pred_len = 96, 48, 96

    model = TestNet(seq_len,pred_len).cuda()
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_func = F.cross_entropy

    train_dataset, vali_dataset = get_dataset(root_path, data_path, seq_len, label_len, pred_len)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    vali_loader = DataLoader(dataset=vali_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)

    device = "cuda"
    total_ep = 10
    for ep in range(total_ep):
        print("Epoch:",ep)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for x, y,x_mark,y_mark in train_loader:
            y = y[:,-pred_len:,:]
            loss, correct_predictions = trainer(x, y, model, optimizer, loss_func, device)
            total_loss += loss
            total_correct += correct_predictions
            total_samples += y[1].shape[0]
        average_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        print(f"Epoch: {ep} Training Loss: {average_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

        average_loss, accuracy = validate(vali_loader, model, loss_func, device)
        print(f"Validation Loss: {average_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
