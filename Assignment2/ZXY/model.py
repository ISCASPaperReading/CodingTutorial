import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, class_dim, image_shape):
        super(Generator, self).__init__()

        self.embed = nn.Embedding(class_dim, class_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + class_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_shape),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embedded_labels = self.embed(labels)
        input_z = torch.cat([z, embedded_labels], dim=1)
        img = self.model(input_z)
        img = img.view(img.size(0), -1)
        return img


class Discriminator(nn.Module):
    def __init__(self, class_dim, image_shape):
        super(Discriminator, self).__init__()

        self.embed = nn.Embedding(class_dim, class_dim)
        self.model = nn.Sequential(
            nn.Linear(image_shape + class_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        embedded_labels = self.embed(labels)
        input_img = torch.cat([img, embedded_labels], dim=1)
        validity = self.model(input_img)
        return validity
