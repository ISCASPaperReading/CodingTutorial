import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

from model import Generator, Discriminator


def main():
    # 参数设置
    latent_dim = 100
    class_dim = 10  # CIFAR-10数据集有10个类别
    img_shape = 3 * 32 * 32  # CIFAR-10图片的形状
    lr = 0.0002
    batch_size = 64
    epochs = 10

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataloader = DataLoader(
        datasets.CIFAR10(root='data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    # 创建生成器和判别器，并定义损失函数和优化器
    generator = Generator(latent_dim, class_dim, img_shape)
    discriminator = Discriminator(class_dim, img_shape)

    adversarial_loss = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # 训练
    for epoch in range(epochs):
        for i, (imgs, label) in enumerate(dataloader):

            # 训练判别器
            valid = torch.ones(imgs.size(0), 1)
            fake = torch.zeros(imgs.size(0), 1)

            real_imgs = imgs.view(imgs.size(0), -1)

            # 训练判别器用真实图片
            discriminator_optimizer.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, label), valid)
            real_loss.backward()

            # 训练判别器用生成图片
            z = torch.randn(imgs.size(0), latent_dim)
            gen_imgs = generator(z, label)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), label), fake)
            fake_loss.backward()
            discriminator_optimizer.step()

            # 训练生成器
            generator_optimizer.zero_grad()
            gen_loss = adversarial_loss(discriminator(gen_imgs, label), valid)
            gen_loss.backward()
            generator_optimizer.step()

            discriminator_loss = real_loss + fake_loss
            generator_loss = gen_loss

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {discriminator_loss.item()}] [G loss: {generator_loss.item()}]")

        # 保存生成的图片

        os.makedirs("images", exist_ok=True)
        save_image(gen_imgs.data[:5], f"images/{epoch}.png", nrow=5, normalize=True)


if __name__ == "__main__":
    main()
