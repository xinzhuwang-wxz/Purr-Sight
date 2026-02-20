"""
手动集成PANNs-CNN14模型核心代码（无第三方库依赖）

来源：https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py

包含：
- ConvBlock: 卷积块类
- Cnn14: PANNs-CNN14模型类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    卷积块：两个卷积层+批归一化+池化
    
    Attributes:
        conv1: 第一个卷积层
        bn1: 第一个批归一化层
        conv2: 第二个卷积层
        bn2: 第二个批归一化层
        relu: ReLU激活函数
    """
    
    def __init__(self, in_channels, out_channels):
        """
        初始化卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        """
        前向传播
        
        Args:
            x: 输入张量
            pool_size: 池化窗口大小，默认(2, 2)
            pool_type: 池化类型，'avg'或'max'，默认'avg'
        
        Returns:
            池化后的特征图
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        return x


class Cnn14(nn.Module):
    """
    PANNs-CNN14模型：6个卷积块+全局池化+全连接层
    
    严格对齐官方实现，输出2048维特征。
    
    Attributes:
        bn0: 输入批归一化层
        conv_block1-6: 6个卷积块
        fc1: 全连接层
        global_pool: 全局平均池化层
    """
    
    def __init__(self):
        """
        初始化CNN14模型
        """
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        前向传播：严格对齐官方实现
        
        Args:
            x: 输入梅尔频谱，形状为(64, 256)或(B, 64, 256)或(B, 1, T, 64)
        
        Returns:
            2048维特征，形状为(B, 2048)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            x = x.transpose(2, 3)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
            x = x.transpose(2, 3)
        elif x.dim() == 4:
            if x.shape[1] == 1:
                x = x.transpose(2, 3)
            elif x.shape[1] == 64:
                x = x.mean(dim=1, keepdim=True)
                x = x.transpose(2, 3)
            else:
                raise ValueError(f"输入通道数应为1，实际为{x.shape[1]}")
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2))
        x = self.conv_block2(x, pool_size=(2, 2))
        x = self.conv_block3(x, pool_size=(2, 2))
        x = self.conv_block4(x, pool_size=(2, 2))
        x = self.conv_block5(x, pool_size=(2, 2))
        x = self.conv_block6(x, pool_size=(1, 1))

        x = self.global_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        return x  