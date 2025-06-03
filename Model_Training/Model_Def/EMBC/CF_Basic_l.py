import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import constant_, kaiming_normal_, trunc_normal_
from timm.layers import DropPath
"""
# 论文题目:SCTNet: Single-Branch CNN with Transformer Semantic Information for Real-Time Segmentation
# 中文题目: 单分支CNN结合Transformer语义信息的实时分割网络
# 论文链接:https://arxiv.org/pdf/2312.17071
# 官方github:https://github.com/xzz777/SCTNet
# 所属机构：华中科技大学人工智能与自动化学院国家多媒体信息智能处理技术重点实验室，美团
# 关键词:实时语义分割,Transformer,单分支CNN,语义信息对齐,深度学习
"""
def constant_init(tensor, val):
    constant_(tensor, val)

def kaiming_init(tensor):
    kaiming_normal_(tensor)

def trunc_normal_init(tensor, mean=0., std=1.):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, drop_rate=0.):
        super(MLP, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.BatchNorm1d(in_channels, eps=1e-06)
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout(drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm1d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv1d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class ConvolutionalAttention(nn.Module):
    def __init__(self, in_channels, inter_channels, num_heads=8):
        super(ConvolutionalAttention, self).__init__()
        assert in_channels % num_heads == 0, \
            "out_channels ({}) should be a multiple of num_heads ({})".format(in_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.BatchNorm1d(in_channels, eps=1e-06)

        # 参数初始化
        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, 7))
        trunc_normal_init(self.kv, std=0.001)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm1d)):
            constant_init(m.weight, val=1.)
            constant_init(m.bias, val=.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)

    def _act_dn(self, x):
        x_shape = x.shape  # (batch, inter_channels, time)
        t = x_shape[2]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])  # (batch, heads, inter_channels//heads, time)
        x = F.softmax(x, dim=3)  # 对时间维度进行归一化
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)  # 归一化后重新缩放
        x = x.reshape([x_shape[0], self.inter_channels, t])  # 恢复形状
        return x

    def forward(self, x):
        x = self.norm(x)  # 归一化
        x1 = F.conv1d(x, self.kv, stride=1, padding=3)  # 卷积操作
        x1 = self._act_dn(x1)
        x1 = F.conv1d(x1, self.kv.transpose(1, 0), stride=1, padding=3)
        return x1

class CFBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, drop_rate=0, drop_path_rate=0, downsample=False):
        super(CFBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.downsample_conv = nn.Conv1d(
                in_channels, 
                in_channels * 2, 
                kernel_size=3, 
                stride=2, 
                padding=1
            )
            self.downsample_norm = nn.BatchNorm1d(in_channels * 2, eps=1e-6)
            in_channels *= 2  # 更新通道数以适配后续模块

        self.attn = ConvolutionalAttention(in_channels, inter_channels=64, num_heads=num_heads)
        self.mlp = MLP(in_channels, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        if self.downsample:
            x = self.downsample_conv(x)
            x = self.downsample_norm(x)
        x_res = x
        x = x_res + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x
    

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
        # 当 stride != 1 时self.conv1 和 self.downsample 层都会对输入进行下采样
        self.conv1 = conv3x1(in_channels, out_channels, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(out_channels, out_channels, stride=1)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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

    def __init__(self, num_BP=1, zero_init_residual=False, norm_layer=nn.BatchNorm1d):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer
        self.input_channels = 64

        # Initial Layers
        self.conv1 = DualDynamicConv1d()
        # self.conv1 = nn.Conv1d(2, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1_block1 = BasicBlock(64, 64, stride=1, norm_layer=norm_layer)
        self.layer1_cf1 = CFBlock(64)
        self.layer1_block2 = BasicBlock(64, 64, stride=1, norm_layer=norm_layer)
        self.layer1_cf2 = CFBlock(64)

        downsample2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=2, bias=False),
            norm_layer(128),
        )
        self.layer2_block1 = BasicBlock(64, 128, stride=2, downsample=downsample2, norm_layer=norm_layer)
        self.layer2_cf1 = CFBlock(64,downsample=True)
        self.layer2_block2 = BasicBlock(128, 128, stride=1, norm_layer=norm_layer)
        self.layer2_cf2 = CFBlock(128)

        downsample3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=2, bias=False),
            norm_layer(256),
        )
        self.layer3_block1 = BasicBlock(128, 256, stride=2, downsample=downsample3, norm_layer=norm_layer)
        self.layer3_cf1 = CFBlock(128,downsample=True)
        self.layer3_block2 = BasicBlock(256, 256, stride=1, norm_layer=norm_layer)
        self.layer3_cf2 = CFBlock(256)

        # Layer 4
        downsample4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, stride=2, bias=False),
            norm_layer(512),
        )
        self.layer4_block1 = BasicBlock(256, 512, stride=2, downsample=downsample4, norm_layer=norm_layer)
        self.layer4_cf1 = CFBlock(256,downsample=True)

        self.layer4_block2 = BasicBlock(512, 512, stride=1, norm_layer=norm_layer)
        self.layer4_cf2 = CFBlock(512)

        # Head Layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_BP)

        # Initialization
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

    def forward(self, x):
        # Initial Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1_block1(x) + self.layer1_cf1(x)
        x = self.layer1_block2(x) + self.layer1_cf2(x)
        x = self.layer2_block1(x) + self.layer2_cf1(x)
        x = self.layer2_block2(x) + self.layer2_cf2(x)
        x = self.layer3_block1(x) + self.layer3_cf1(x)
        x = self.layer3_block2(x) + self.layer3_cf2(x)
        x = self.layer4_block1(x) + self.layer4_cf1(x)
        x = self.layer4_block2(x) + self.layer4_cf2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def Resnet18_1D():
    return ResNet()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.randn(32, 32, 625).to(device)  # (batch_size, channels, time_steps)
    print("输入形状:", input.shape)

    # 不进行下采样
    cfb_no_down = CFBlock(32,downsample=False).to(device)
    output_no_down = cfb_no_down(input)
    print("不下采样输出形状:", output_no_down.shape)
    
    # 进行下采样
    cfb_down = CFBlock(32,downsample=True).to(device)
    output_down = cfb_down(input)
    print("下采样输出形状:", output_down.shape)