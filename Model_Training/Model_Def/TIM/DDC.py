import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
"""
消融实验之DDC
"""
class MSEBlock(nn.Module): #  SE with Attention on Multiple Scales
    def __init__(self, in_channels, reduction=4):
        super(MSEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, L = x.shape
        avg_pool = self.global_avg_pool(x).view(B, C)
        max_pool = self.global_max_pool(x).view(B, C)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        weights = self.fc(pooled).view(B, C, 1)
        return x * weights

class DynamicConv1d(nn.Module):
    """Dynamic Convolution for 1D time series without reduction_ratio."""
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=1, bias=True):
        super().__init__()
        self.num_groups = num_groups
        self.K = kernel_size
        self.weight = nn.Parameter(torch.randn(num_groups, out_channels, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool1d(kernel_size)
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * num_groups, kernel_size=1),
            nn.BatchNorm1d(out_channels * num_groups),
            nn.GELU()
        )
        self.bias = nn.Parameter(torch.randn(num_groups, out_channels), requires_grad=True) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, L = x.shape
        pooled = self.pool(x)
        proj_output = self.proj(pooled)
        expected_channels = self.num_groups * C
        assert proj_output.shape[1] % expected_channels == 0, \
            f"proj_output channels {proj_output.shape[1]} must be divisible by {expected_channels}."

        reshaped_output = proj_output.view(B, self.num_groups, -1, self.K)

        scale = torch.softmax(reshaped_output, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = weight.sum(dim=1).reshape(-1, 1, self.K)
        x = F.conv1d(x.reshape(1, -1, L), weight, padding=self.K // 2, groups=B * C)
        x = x.reshape(B, -1, L)
        return x

class DualDynamicConv1d(nn.Module):
    def __init__(self, kernel_size=3, num_groups=8, bias=True, reduction=4):
        super().__init__()
        # 每个通道独立的动态卷积
        self.independent_conv1 = DynamicConv1d(1, 32, kernel_size, num_groups, bias)
        self.independent_conv2 = DynamicConv1d(1, 32, kernel_size, num_groups, bias)

        # 联合动态卷积
        self.joint_conv = DynamicConv1d(2, 64, kernel_size, num_groups, bias)
        self.mse_block = MSEBlock(64, reduction)
        # 用于融合的可学习权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 独立特征的权重
        self.beta = nn.Parameter(torch.tensor(0.5))   # Weight for joint features

        # 最终标准化和激活
        self.norm = nn.BatchNorm1d(64)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, L = x.shape
        assert C == 2, "Input must have 2 channels (PPG and ECG)."
        # 每个通道独立的动态卷积
        x1 = self.independent_conv1(x[:, 0:1, :])  # Shape: (B, 16, L)
        x2 = self.independent_conv2(x[:, 1:2, :])  # Shape: (B, 16, L)
        # 拼接独立特征
        independent_features = torch.cat([x1, x2], dim=1)  # Shape: (B, 32, L)
        # 联合动态卷积
        joint_features = self.joint_conv(x)  # Shape: (B, 32, L)
        joint_features = self.mse_block(joint_features)  # Apply MSE module
        # 归一化融合权重
        alpha_normalized = torch.sigmoid(self.alpha)  # Scale to [0, 1]
        beta_normalized = torch.sigmoid(self.beta)    # Scale to [0, 1]
        # 熔断器独立和联合功能
        fused_features = alpha_normalized * independent_features + beta_normalized * joint_features
        # 应用标准化和激活
        fused_features = self.norm(fused_features)
        fused_features = self.activation(fused_features)
        return fused_features

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
  
class BasicBlock(nn.Module): #BasicBlock 始终具有 dilation=1 和 groups=1
    expansion: int = 1 #基本块期望输入和输出具有相同数量的通道
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None, norm_layer = nn.BatchNorm1d,):
        super(BasicBlock,self).__init__()
        # 当 stride != 1 时，self.conv1 和 self.downsample 层都会对输入进行下采样
        self.conv1 = conv3x1(in_channels, out_channels, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(out_channels, out_channels, stride=1)
        self.bn2 = norm_layer(out_channels)
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

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_BP=1, zero_init_residual=False, norm_layer=nn.BatchNorm1d):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer
        self.input_channels = 64
        self.conv1 = DualDynamicConv1d()
        # self.conv1 = nn.Conv1d(2, self.input_channels, kernel_size=7, stride=2, padding=3,
        #                        bias=False) # 2CH -> 64CH，现在Len->Len/2
        self.bn1 = norm_layer(self.input_channels)
        self.relu = nn.ReLU(inplace=True) 
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) #现在仅>仅/4
        self.layer1 = self._make_layer(block, out_channels=64, num_blocks=layers[0]) # 64CH->64CH, NowLen=Len/4
        self.layer2 = self._make_layer(block, out_channels=128, num_blocks=layers[1], stride=2) # 64CH->128CH, NowLen=Len/8
        self.layer3 = self._make_layer(block, out_channels=256, num_blocks=layers[2], stride=2) # 128CH->256CH, NowLen=Len/16
        self.layer4 = self._make_layer(block, out_channels=512, num_blocks=layers[3], stride=2) # 256CH->512CH, NowLen=Len/32
        self.avgpool = nn.AdaptiveAvgPool1d(1) # 最终特征图=512*1
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

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
         # 调整恒等映射方法以匹配所需的通道数和长度   
        if stride != 1 or self.input_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.input_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )
        layers = []
        # 第一个块使输入通道适应输出通道，并对长度进行下采样 
        layers.append(block(self.input_channels, out_channels, stride, downsample, norm_layer))
        # 下次调用_make_layer时，输入通道是之前的输出通道
        self.input_channels = out_channels * block.expansion
        # 其余块不改变长度或通道
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
    
def DDC34():
    return ResNet(block=BasicBlock, layers=[3,4,6,3])