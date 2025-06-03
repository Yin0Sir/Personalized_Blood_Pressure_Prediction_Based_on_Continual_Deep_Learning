import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
"""
(batch_size, channels, sequence_length)
"""

class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.se_ratio = se_ratio

        # 挤压部分：压缩特征图的维数
        self.squeeze = nn.AdaptiveAvgPool1d(1)

        # 激励部分：生成特征图权重
        self.excitation = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // self.se_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels // self.se_ratio, self.in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x).permute(0, 2, 1)  # 压缩特征图
        out = self.excitation(out).permute(0, 2, 1)  # 生成特征图权重
        return x * out.expand_as(x)  # 将权重应用于特征图

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.se_block = SEBlock(out_channels, se_ratio=16)  # 添加 SE 模块

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se_block(out)  # SE 模块应用程序
        out += identity
        out = self.relu(out)
        return out

class SEResNet(nn.Module):
    def __init__(self, block, layers, num_BP=1):
        super(SEResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):

        # 处理数据流
        out = self.conv1(x.unsqueeze(1))
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        x = torch.flatten(out, 1)
        
        return x

class CMA_Block(nn.Module):
    def __init__(self, channel):
        super(CMA_Block, self).__init__()

        self.scale = channel ** -0.5  # 缩放因子

    def forward(self, ppg_features, ecg_features):
        # ppg_features 和 ecg_features 的形状为 (batch_size, channel)

        q = ppg_features  # 使用 PPG 特征作为查询 (query)
        k = ecg_features  # 使用 ECG 特征作为键 (key)
        v = ecg_features  # 使用 ECG 特征作为值 (value)

        # 计算查询和键之间的相似度（点积）
        attn = torch.matmul(q, k.T) * self.scale  # (batch_size, num_features) × (batch_size, num_features).T -> (batch_size, batch_size)
        
        # softmax 归一化
        attn = attn.softmax(dim=-1)  # 对每个查询进行 softmax，归一化为概率分布 (batch_size, batch_size)

        # 计算加权求和（将注意力权重与值进行加权）
        z = torch.matmul(attn, v)  # (batch_size, batch_size) × (batch_size, num_features) -> (batch_size, num_features)

        # 输出融合后的特征
        return z
    
class DualStreamNetwork(nn.Module):
    def __init__(self, net, cma_block):
        super(DualStreamNetwork, self).__init__()
        self.net = net  # 流网络
        self.cma_block = cma_block  # CMA_Block 用于融合
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        # 输入数据形状 (batch_size, 2, sequence_length)
        ppg_input = x[:, 0, :]  # PPG 通道
        ecg_input = x[:, 1, :]  # ECG 通道

        # 分别通过两个流网络提取特征
        ppg_features = self.net(ppg_input)  # 提取 PPG 特征
        ecg_features = self.net(ecg_input)  # 提取 ECG 特征

        # 将提取的特征传递给 CMA_Block 进行融合
        output = self.cma_block(ppg_features, ecg_features)
        x = self.fc(output)

        return x

def DSANet18():
    return DualStreamNetwork(SEResNet(block=ResidualBlock, layers= [2, 2, 2, 2]), CMA_Block(channel=512))

def DSANet34():
    return DualStreamNetwork(SEResNet(block=ResidualBlock, layers= [3, 4, 6, 3]), CMA_Block(channel=512))

if __name__ == '__main__':
    
    model = DSANet18()

    # 假设有一个输入数据
    input_data = torch.randn(32, 2, 1250)  # (batch_size, channels, sequence_length)
    output = model(input_data)
    print("Output shape:", output.shape)