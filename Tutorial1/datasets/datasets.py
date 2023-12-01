from torch.utils.data import Dataset
from datasets.preprocess import mnist_transform
from torchvision.datasets import MNIST


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = [0 for i in range(100)]

    def __getitem__(self, index):
        return self.data[index]
    

def get_mnist_dataset():
    train_dataset = MNIST(root='./data', train=True, transform=mnist_transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=mnist_transform, download=True)
    return train_dataset, test_dataset