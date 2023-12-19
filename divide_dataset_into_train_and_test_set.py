# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:30:00 2023
数据集分割为训练集和测试集
@author: 46764
"""

import os
import shutil
import random

# 设置数据集路径
dataset_path = './archive/images'
train_path = './archive/train'
test_path = './archive/test'

# 创建训练集和测试集目录
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# 设置分割比例（例如，80% 训练集，20% 测试集）
split_ratio = 0.8

# 获取数据集中的文件列表
file_list = os.listdir(dataset_path)

# 随机打乱文件列表
random.shuffle(file_list)

# 计算划分的索引
split_index = int(len(file_list) * split_ratio)

# 将文件移动到训练集目录
for file in file_list[:split_index]:
    src_path = os.path.join(dataset_path, file)
    dst_path = os.path.join(train_path, file)
    shutil.copy(src_path, dst_path)

# 将文件移动到测试集目录
for file in file_list[split_index:]:
    src_path = os.path.join(dataset_path, file)
    dst_path = os.path.join(test_path, file)
    shutil.copy(src_path, dst_path)

print("数据集已成功分割为训练集和测试集。")
