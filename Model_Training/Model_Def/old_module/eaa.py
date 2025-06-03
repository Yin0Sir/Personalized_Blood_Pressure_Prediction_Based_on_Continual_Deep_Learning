import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import einops

def conv3x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    """3x1 convolution with padding, output_len=input_len"""
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1, #如果 dilation =n，则内核大小相当于 3+2n。要保持相同的输出大小，请使用 padding=dilation。
        groups=1,
        bias=False,
        dilation=1,
    )

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution with no padding, output_len=input_len """
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )

class EfficientAdditiveAttention(nn.Module):

    def __init__(self, in_dims, token_dim, num_heads=1):
        super().__init__()
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD
        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1
        A = torch.nn.functional.normalize(A, dim=1) # BxNx1
        G = torch.sum(A * query, dim=1) # BxD
        G = einops.repeat(G, "b d -> b repeat d", repeat=key.shape[1]) # BxNxD
        out = self.Proj(G * key) + query #BxNxD
        out = self.final(out) # BxNxD

        return out

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=nn.BatchNorm1d, attention=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(in_channels, out_channels, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(out_channels, out_channels, stride=1)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.attention = EfficientAdditiveAttention(in_dims=out_channels, token_dim=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.attention is not None:
            B, D, N = out.shape
            out = out.permute(0, 2, 1)
            out = self.attention(out)
            out = out.permute(0, 2, 1)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_BP=1, zero_init_residual=False, norm_layer=nn.BatchNorm1d):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer
        self.input_channels = 64

        self.conv1 = nn.Conv1d(2, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, attention=True)  # 添加注意力
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride=1, attention=False):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.input_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.input_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )

        layers = [block(self.input_channels, out_channels, stride, downsample, norm_layer, attention=attention)]
        self.input_channels = out_channels * block.expansion
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

def Resnet18_1D():
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

def Resnet34_1D():
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3])
