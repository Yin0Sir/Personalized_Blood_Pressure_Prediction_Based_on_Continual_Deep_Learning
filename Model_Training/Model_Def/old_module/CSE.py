import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 论文：Correlation Networks for Extreme Multi-label Text Classification

ACT2FN = {'elu': F.elu, 'relu': F.relu, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh}

class CorNetBlock(nn.Module):
    def __init__(self, context_size, output_size, cornet_act='sigmoid', **kwargs):
        super(CorNetBlock, self).__init__()
        self.dstbn2cntxt = nn.Linear(output_size, context_size)
        self.cntxt2dstbn = nn.Linear(context_size, output_size)
        self.act_fn = ACT2FN[cornet_act]

    def forward(self, output_dstrbtn):
        identity_logits = output_dstrbtn
        output_dstrbtn = self.act_fn(output_dstrbtn)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        context_vector = F.elu(context_vector)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        output_dstrbtn = output_dstrbtn + identity_logits
        return output_dstrbtn

class CorNet(nn.Module):
    def __init__(self, output_size, cornet_dim=100, n_cornet_blocks=2, **kwargs):
        super(CorNet, self).__init__()
        self.intlv_layers = nn.ModuleList(
            [CorNetBlock(cornet_dim, output_size, **kwargs) for _ in range(n_cornet_blocks)])
        for layer in self.intlv_layers:
            nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)

    def forward(self, logits):
        for layer in self.intlv_layers:
            logits = layer(logits)
        return logits
    
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

        self.conv1 = nn.Conv1d(2, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(512, num_BP)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

class CombinedNet(nn.Module):
    def __init__(self, cornet_output_size=512, cornet_dim=1000, n_cornet_blocks=2, num_BP=1):
        super(CombinedNet, self).__init__()
        
        # 将 ResNet 初始化为特征提取器
        self.resnet = SEResNet(block=ResidualBlock, layers=[2, 2, 2, 2], num_BP=num_BP)
        # 初始化 CorNet 以进行上下文增强
        self.cornet = CorNet(output_size=cornet_output_size, cornet_dim=cornet_dim, n_cornet_blocks=n_cornet_blocks)
        # 添加最终线性层以将 CorNet 输出映射到单个 BP 预测
        self.final_fc = nn.Linear(cornet_output_size, num_BP)

    def forward(self, x):
        # 第 1 步：将输入传递给 ResNet 进行特征提取
        resnet_features = self.resnet.forward(x)  # 应用 ResNet 特征提取器
        # 步骤 2：使用 CorNet 增强和融合 ResNet 特征中的上下文
        cornet_output = self.cornet(resnet_features)
        # 步骤 3：将 CorNet 输出映射到最终 BP 预测
        output = self.final_fc(cornet_output)
        
        return output

def CSE_Net():
    return CombinedNet(cornet_output_size=512, cornet_dim=1000, n_cornet_blocks=2, num_BP=1)

# Example usage
if __name__ == '__main__':
    input = torch.rand(32, 2, 5000)  # Example input tensor with (batch_size, channels, sequence_length)
    model = CombinedNet(cornet_output_size=512, cornet_dim=1000, n_cornet_blocks=2, num_BP=1)
    output = model(input)
    print("Output shape:", output.shape)  # Expected shape: (batch_size, num_BP), e.g., (10, 1)
