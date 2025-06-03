import torch
import torch.nn as nn
from torch import Tensor

"""
(batch_size, channels, sequence_length)
"""

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

class Bottleneck(nn.Module):
    expansion = 4  # Bottleneck 的输出通道是输入通道的 4 倍
    def __init__(self, in_channels, med_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, med_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(med_channels)
        self.conv2 = nn.Conv1d(med_channels, med_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(med_channels)
        self.conv3 = nn.Conv1d(med_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
    
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
        self.fc = nn.Linear(512, num_BP)

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
        x = self.fc(x)
        return x

class ResNet_Bottleneck(nn.Module):
    def __init__(self, block, layers, in_channels=2, output_dim=1, zero_init_residual=False):
        super(ResNet_Bottleneck, self).__init__()
        self.in_channels = 64  # 初始通道数
        self.conv1 = nn.Conv1d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # 构建每一层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, output_dim)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, out_channels * block.expansion, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, out_channels * block.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
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
    
def Resnet18_1D():
    return ResNet(block=BasicBlock, layers=[2,2,2,2])
def Resnet34_1D():
    return ResNet(block=BasicBlock, layers=[3,4,6,3])
def ResNet50():
    return ResNet_Bottleneck(block=Bottleneck, layers=[3, 4, 6, 3])
def SEResnet18_1D():
    return SEResNet(block=ResidualBlock, layers=[2,2,2,2])
def SEResnet34_1D():
    return SEResNet(block=ResidualBlock, layers=[3,4,6,3])