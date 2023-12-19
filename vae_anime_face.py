# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:03:45 2023

@author: 46764
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from PIL import Image

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # 均值和方差的全连接层
        self.fc_mu = nn.Linear(256 * 4 * 4, 200)
        self.fc_logvar = nn.Linear(256 * 4 * 4, 200)

        # 解码器
        self.decoder_fc = nn.Sequential(
            nn.Linear(200, 256 * 4 * 4),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        # print(z.shape)
        # print(z.view(z.size(0), -1, 1, 1).shape)
        z = self.decoder_fc(z)
        # print(z.shape)
        z = self.decoder_conv(z.view(z.size(0), 256, 4, 4))
        return z, mu, logvar

# 定义训练函数
def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# 定义测试函数
def test(epoch):
    model.eval()  # 设置模型为评估模式
    test_loss = 0

    with torch.no_grad():  # 不需要计算梯度
        for i, data in enumerate(test_loader):  # 遍历测试数据集的批次
            data = data.to(device)  # 将数据移动到设备（CPU或GPU）上
            recon_batch, mu, logvar = model(data)  # 使用模型进行前向传播
            test_loss += loss_function(recon_batch, data, mu, logvar).item()  # 计算测试损失

            if i == 0:
                # 保存一些图像用于比较（原始图像和模型生成的图像）
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(batch_size, 3, 64, 64)[:n]])
                save_image(comparison.cpu(), './archive/output/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)  # 计算平均测试损失
    print('====> Test set loss: {:.4f}'.format(test_loss))


# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3, 64, 64), reduction='sum')

    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.input_filenames = sorted([f for f in os.listdir(root) if f.endswith('.jpg')])

    def __getitem__(self, index):
        input_img_path = os.path.join(self.root, self.input_filenames[index])
        input_img = Image.open(input_img_path).convert('RGB')  # Convert to RGB
        if self.transform:
            input_img = self.transform(input_img)
        return input_img  

    def __len__(self):
        return len(self.input_filenames)
        


# 设置超参数和设备
batch_size = 64
epochs = 10
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# 创建自定义数据集实例
train_dataset = CustomDataset(root='./archive/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(root='./archive/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 初始化模型和优化器
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

# 保存模型
torch.save(model.state_dict(), 'vae_anime_model.pth')
