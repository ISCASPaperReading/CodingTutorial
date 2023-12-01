import torchvision.transforms as T

mnist_transform = T.Compose([
    T.ToTensor(),  # 将图像转换为张量
    T.Normalize((0.5,), (0.5,))  # 标准化张量，使其范围在[-1, 1]之间
])