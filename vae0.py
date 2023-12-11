# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:36:32 2023
写一个变分自编码器（VAE）
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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, input_folder, target_folder, transform=None):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.transform = transform
        
        self.input_filenames = sorted(os.listdir(input_folder))
        self.target_filenames = sorted(os.listdir(target_folder))
        
    def __len__(self):
        return len(self.input_filenames)
    
    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_folder, self.input_filenames[idx])
        target_img_path = os.path.join(self.target_folder, self.target_filenames[idx])
        
        input_img = Image.open(input_img_path).convert('RGB')  # Convert to RGB
        target_img = Image.open(target_img_path).convert('RGB')  # Convert to RGB
        
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        
        return input_img, target_img



# 定义卷积自编码神经网络
class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        #encoder
        self.up_dim = nn.Sequential(
            nn.Conv2d( in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU()
            )
        self.drop_dim = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3),
            nn.ReLU()
            )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # 最大池化为了减少全连接层的参数量
        
        self.fc1 = nn.Sequential(nn.Linear(4*126*126, 512),
                                 nn.ReLU()
                                 )
        self.fc2_mean = nn.Linear(512, latent_size)
        self.fc2_logvar = nn.Linear(512, latent_size)
        #decoder
        self.fc3 = nn.Sequential(
            nn.Linear(latent_size, 4*252*252),
            nn.ReLU()
            )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3),
            # nn.ReLU(),#生成模型最后一层不应该激活
            )
    def encode(self,x):
        x = self.up_dim(x)
        x = self.drop_dim(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 4, 252, 252)
        # print(z.shape)
        x_recon = self.decoder(z)
        x_recon = torch.sigmoid(x_recon) 
        return x_recon
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar

def loss_function(x_recon, target_imgs, mean, logvar):
    # Reconstruction loss
    recon_loss = nn.functional.binary_cross_entropy(x_recon, target_imgs.to(device), reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + kl_loss
    return total_loss

latent_size = 20
vae = VAE(latent_size)
vae = vae.to(device)
print(vae)

optimizer = optim.Adam(vae.parameters(), lr=0.001)

      
# 输入文件夹路径和输出文件夹路径
input_folder  = r"D:\F\MATLAB\SARdata\s1_1"
target_folder  = r"D:\F\MATLAB\SARdata\s2_1"
output_folder = r"D:\F\MATLAB\SARdata\VAEs12s2_1"
# 创建输出文件夹，如果它不存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
dataset = CustomDataset(input_folder, target_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
  
        
        

# 训练模型
# torch.autograd.set_detect_anomaly(True)#开启异常检测

num_epochs = 60
for epoch in range(num_epochs):
    vae.train()  # 设置模型为训练模式,这主要是针对一些在训练和测试时行为不同的层，比如Batch Normalization和Dropout。
    for input_imgs, target_imgs in data_loader:
        optimizer.zero_grad()
        
        # 前向传播
        x = input_imgs
        x = x.to(device)
        x_recon, mean, logvar = vae(x)
        loss = loss_function(x_recon, target_imgs, mean, logvar)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        
        


# 保存输出图片到指定文件夹
os.makedirs(output_folder, exist_ok=True)

with torch.no_grad():
    for i, (input_imgs, _) in enumerate(data_loader):
        input_output ,_ ,_ = vae(input_imgs.to(device))
        for j in range(input_output.size(0)):
            output_img = input_output[j].permute(1, 2, 0).numpy() * 255
            output_img_path = os.path.join(output_folder, f'output_{i * data_loader.batch_size + j}.png')
            Image.fromarray(output_img.astype('uint8')).save(output_img_path)

print('Output images saved to:', output_folder)



        
    