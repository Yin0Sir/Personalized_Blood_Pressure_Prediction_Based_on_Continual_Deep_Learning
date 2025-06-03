import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

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
        padding=1,  # 如果 dilation =n，则内核大小相当于 3+2n。要保持相同的输出大小，请使用 padding=dilation。
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
        logits = logits.permute(0, 2, 1)
        for layer in self.intlv_layers:
            logits = layer(logits)
        logits = logits.permute(0, 2, 1)
        return logits

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None, norm_layer = nn.BatchNorm1d):
        super(BasicBlock, self).__init__()
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

class CRNet(nn.Module):
    def __init__(self, block, layers, num_BP=1, zero_init_residual=False, norm_layer=nn.BatchNorm1d):
        super(CRNet, self).__init__()
        self._norm_layer = norm_layer

        self.input_channels = 64
        self.conv1 = nn.Conv1d(2, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, out_channels=64, num_blocks=layers[0])
        self.layer2 = self._make_layer(block, out_channels=128, num_blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, out_channels=256, num_blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, out_channels=512, num_blocks=layers[3], stride=2)
        
        # Add CorNet modules after each layer
        self.cornet1 = CorNet(output_size=64, cornet_dim=128)
        self.cornet2 = CorNet(output_size=128, cornet_dim=256)
        self.cornet3 = CorNet(output_size=256, cornet_dim=512)
        self.cornet4 = CorNet(output_size=512, cornet_dim=1024)

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

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
            
        if stride != 1 or self.input_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.input_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.input_channels, out_channels, stride, downsample, norm_layer))
        self.input_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.input_channels, out_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Process each layer + CorNet block
        x = self.layer1(x)
        x = self.cornet1(x)
        x = self.layer2(x)
        x = self.cornet2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cornet4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
def RCNet18():
    return CRNet(block=BasicBlock, layers=[2, 2, 2, 2])

def RCNet34():
    return CRNet(block=BasicBlock, layers=[3, 4, 6, 3])

if __name__ == '__main__':
    input = torch.rand(32, 2, 1250)  # Example input tensor with (batch_size, channels, sequence_length)
    model = RCNet18()
    output = model(input)
    print("Output shape:", output.shape)  # Expected shape: (batch_size, num_BP), e.g., (10, 1)
