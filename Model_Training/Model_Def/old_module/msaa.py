import torch
import torch.nn as nn
from torch import Tensor

class ChannelAttentionModule1D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule1D, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FusionConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv1D, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv1d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv1d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv1d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule1D()
        self.channel_attention = ChannelAttentionModule1D(dim)
        self.up = nn.Conv1d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv1d(in_channels, dim, kernel_size=1, stride=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x):
        x_fused = self.down(x)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return self.leaky_relu(x_out)

class MSAA1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA1D, self).__init__()
        self.fusion_conv = FusionConv1D(in_channels, out_channels)

    def forward(self, x):
        x_fused = self.fusion_conv(x)
        return x_fused

def conv3x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    """3x1 convolution with padding, output_len=input_len"""
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=1,
        bias=False,
        dilation=1,
    )

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution with no padding, output_len=input_len"""
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )

class BasicBlockWithMSAA(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=nn.BatchNorm1d):
        super(BasicBlockWithMSAA, self).__init__()

        # ������������������ BN ��
        self.conv1 = conv3x1(in_channels, out_channels, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(out_channels, out_channels, stride=1)
        self.bn2 = norm_layer(out_channels)

        # ���� MSAA1D ����
        self.msaa = MSAA1D(out_channels, out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.msaa(identity)

        out += identity
        out = self.relu(out)

        return out

class ResNetWithMSAA(nn.Module):
    def __init__(self, block, layers, num_BP=1, zero_init_residual=False, norm_layer=nn.BatchNorm1d):
        super(ResNetWithMSAA, self).__init__()
        self._norm_layer = norm_layer
        self.input_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(2, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.input_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 初始池化层
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet 各层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局池化和输出层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_BP)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockWithMSAA):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        # 下采样以匹配特征图大小
        if stride != 1 or self.input_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.input_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.input_channels, out_channels, stride, downsample, norm_layer))
        self.input_channels = out_channels * block.expansion
        
        # 生成后续层
        for _ in range(1, num_blocks):
            layers.append(block(self.input_channels, out_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
# 创建使用 MSAA1D 作为跳跃连接的 ResNet18 版本
def Resnet18_1D_with_MSAA():
    return ResNetWithMSAA(block=BasicBlockWithMSAA, layers=[2, 2, 2, 2])
