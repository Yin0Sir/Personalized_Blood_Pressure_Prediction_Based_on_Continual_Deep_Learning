import torch
import torch.nn as nn
from torch import Tensor

"""
(batch_size, channels, sequence_length)
"""

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool1d(1)  # 时间维度进行平均池化
        self.pool_l = nn.AdaptiveAvgPool1d(1)  # 沿时间轴池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1)
        self.conv3x3 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, l = x.size()  # 输入形状为 (b, c, l)，b: batch size, c: 通道数, l: 长度
        group_x = x.reshape(b * self.groups, -1, l)  # (b*g, c//g, l)
        x_l = self.pool_l(group_x)  # 在长度维度进行池化
        hw = self.conv1x1(x_l)  # 通过 1x1 卷积融合处理
        x_l = hw.sigmoid()  # 对输出应用 Sigmoid 激活
        x1 = self.gn(group_x * x_l)  # 使用 GroupNorm 进行归一化处理
        x2 = self.conv3x3(group_x)  # 使用 3x3 卷积处理
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 全局池化并应用 softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 调整 x2 的形状, (b*g, c//g, t)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 使用全局池化和 softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 调整 x1 的形状， (b*g, c//g, t)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, l)  # 计算权重
        return (group_x * weights.sigmoid()).reshape(b, c, l)  # 返回加权后的特征图

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
        
        self.conv1 = nn.Conv1d(2, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False) # 2CH -> 64CH，现在Len->Len/2
        self.bn1 = norm_layer(self.input_channels)
        self.relu = nn.ReLU(inplace=True) 
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) #现在仅>仅/4
        
        self.layer1 = self._make_layer(block, out_channels=64, num_blocks=layers[0]) # 64CH->64CH, NowLen=Len/4
        self.layer2 = self._make_layer(block, out_channels=128, num_blocks=layers[1], stride=2) # 64CH->128CH, NowLen=Len/8
        self.layer3 = self._make_layer(block, out_channels=256, num_blocks=layers[2], stride=2) # 128CH->256CH, NowLen=Len/16
        self.layer4 = self._make_layer(block, out_channels=512, num_blocks=layers[3], stride=2) # 256CH->512CH, NowLen=Len/32
        self.ema1 = EMA(64)
        self.ema2 = EMA(128)
        self.ema3 = EMA(256)
        self.ema4 = EMA(512)
    
        self.avgpool = nn.AdaptiveAvgPool1d(1) # 最终特征图=512*1
        self.fc = nn.Linear(512 * block.expansion, num_BP)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 对每个残差分支中的最后一个BN进行零初始化，
        # 使得残差分支从零开始，并且每个残差块的行为就像一个恒等式。
        # 根据 https://arxiv.org/abs/1706.02677，这将模型改进了 0.2~0.3%
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
        # 请参阅注释 [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        X = self.ema1(x)
        x = self.layer2(x)
        X = self.ema2(x)
        x = self.layer3(x)
        X = self.ema3(x)
        x = self.layer4(x)
        X = self.ema4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
def Resnet18_ema():
    return ResNet(block=BasicBlock, layers=[2,2,2,2])
def Resnet34_ema():
    return ResNet(block=BasicBlock, layers=[3,4,6,3])

# 测试 EMA1D 模块
if __name__ == '__main__':
    block = EMA(64).cuda()
    input = torch.rand(1, 64, 128).cuda()  # 输入大小为 (1, 64, 128)，即 1 个样本，64 个通道，长度 128
    output = block(input)
    print(input.size(), output.size())
