# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:57:05 2023
用我搓的vae网络试试生成数字图片
@author: 46764
"""


import torch
import torch.nn as nn
import torch.optim as optim
import os


import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt


# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 定义卷积自编码神经网络
class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        #encoder
        self.up_dim = nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=8, kernel_size=3),
            nn.ReLU()
            )
        self.drop_dim = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3),
            nn.ReLU()
            )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # 最大池化为了减少全连接层的参数量
        self.embedding = nn.Embedding(10, 4*12*12)

        self.fc1 = nn.Sequential(nn.Linear(4*12*12, 256),
                                 nn.ReLU()
                                 )

        self.fc2_mean = nn.Linear(256, latent_size)
        self.fc2_logvar = nn.Linear(256, latent_size)
        #decoder
        self.fc3 = nn.Sequential(
            nn.Linear(latent_size + 1, 4*24*24),
            nn.ReLU()
            )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3),
            # nn.ReLU(),#生成模型最后一层不应该激活
            )
    def encode(self,x, label):
        x = self.up_dim(x)
        print(x.shape)

        x = self.drop_dim(x)
        print(x.shape)

        x = self.maxpool(x)
        print(x.shape)

        x = x.view(x.size(0), -1)
        print(x.shape)
        label_embedding = self.embedding(label)
        print(x.shape, label_embedding.shape)
        x = self.fc1(x + label_embedding)

        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z, label):
        print(z.shape, label.shape)
        z = torch.cat([z, label.unsqueeze(1)], dim=1)
        print(z.shape)
        z = self.fc3(z)
        z = z.view(z.size(0), 4, 24, 24)
        print(z.shape)
        x_recon = self.decoder(z)
        print(z.shape)
        x_recon = torch.sigmoid(x_recon) 
        return x_recon
    
    def forward(self, x, label):
        mean, logvar = self.encode(x, label)
        z = self.reparameterize(mean, logvar)
        
        x_recon = self.decode(z, label)
        return x_recon, mean, logvar


class CVAE(nn.Module):
    def __init__(self, latent_size, num_classes):
        super(CVAE, self).__init__()
        #encoder
        self.up_dim = nn.Sequential(
            nn.Conv2d( in_channels=(1+num_classes), out_channels=8, kernel_size=3),
            nn.ReLU()
            )
        self.drop_dim = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3),
            nn.ReLU()
            )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # 最大池化为了减少全连接层的参数量

        self.fc1 = nn.Sequential(nn.Linear(4*12*12, 256),
                                 nn.ReLU()
                                 )

        self.fc2_mean = nn.Linear(256, latent_size)
        self.fc2_logvar = nn.Linear(256, latent_size)
        #decoder
        self.fc3 = nn.Sequential(
            nn.Linear(latent_size + num_classes, 4*24*24),
            nn.ReLU()
            )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3),
            # nn.ReLU(),#生成模型最后一层不应该激活
            )
    def encode(self,x, y):
        # 将独热编码的标签扩展为与图像相同的空间尺寸
        y_expanded = y.view(y.size(0), y.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        y_expanded.to(device)
        # 在通道维度上拼接输入图像和标签
        xy = torch.cat([x, y_expanded], dim=1)
        
        x = self.up_dim(xy)
        print(x.shape)
        x = self.drop_dim(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc1(x)

        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z, y):
        print(z.shape, y.shape)
        zy = torch.cat([z, y], dim=1)
        # print(z.shape)
        z = self.fc3(zy)
        z = z.view(z.size(0), 4, 24, 24)
        # print(z.shape)
        x_recon = self.decoder(z)
        # print(z.shape)
        x_recon = torch.sigmoid(x_recon) 
        return x_recon
    
    def forward(self, x, y):
        mean, logvar = self.encode(x, y)
        z = self.reparameterize(mean, logvar)
        
        x_recon = self.decode(z, y)
        return x_recon, mean, logvar
def loss_function(x_recon, x, mean, logvar):
    # Reconstruction loss
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + kl_loss
    return total_loss

latent_size = 20
BATCH_SIZE=100
num_classes=10
vae = CVAE(latent_size, num_classes)
# vae = VAE(latent_size, num_classes)

vae = vae.to(device)
print(vae)

optimizer = optim.Adam(vae.parameters(), lr=0.001)
import torchvision.datasets as datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data_MNIST', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 迭代 DataLoader
num_epochs = 100
for epoch in range(num_epochs):
    vae.train()  # 设置模型为训练模式,这主要是针对一些在训练和测试时行为不同的层，比如Batch Normalization和Dropout。
    for batch in train_loader:
        data, label = batch
        # 将标签转换为独热编码
        labels_one_hot = nn.functional.one_hot(label, num_classes=num_classes).float()

        optimizer.zero_grad()
        
        # 前向传播
        x = data
        x = x.to(device)
        
        x_recon, mean, logvar = vae(x, labels_one_hot.to(device))
        loss = loss_function(x_recon, x, mean, logvar)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    
output_folder = r"D:\F\DL\AE\vae\MNISTvae_result"


# 使用训练好的 VAE 生成不同数字的图片
vae.eval()
with torch.no_grad():
    # 生成一个随机噪声向量，并指定标签
    z = torch.randn(1, latent_size).to(device)  # latent_dim 替换为您的潜在变量维度
    
    labels_one_hot = nn.functional.one_hot(torch.tensor(1), num_classes=num_classes).float()
    labels_one_hot = torch.unsqueeze(labels_one_hot.to(device), 0)


    generated_image = vae.decode(z,  labels_one_hot)
    
# 将张量的值范围从[-1, 1]映射到[0, 255]
image = (generated_image.squeeze()) * 255

# 转换为numpy数组
image_np = image.cpu().detach().numpy().astype(int)

# 显示灰度图
plt.imshow(image_np, cmap='gray', vmin=0, vmax=255)
plt.show()


# torch.save({
#     'epoch': epoch,  # 如果你想保存当前训练的轮次
#     'model_state_dict': vae.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     # 可以添加其他你想保存的信息，比如训练损失等
# }, './generator_of_vae_MNIST.pth')



# # 加载模型
# checkpoint = torch.load('./generator_of_vae_MNIST.pth')
# vae.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# # 如果你保存了轮次信息，你也可以加载它
# epoch = checkpoint['epoch']

