import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=4):
#         super(SEBlock, self).__init__()
#         self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         B, C, L = x.shape
#         pooled = self.global_pool(x).view(B, C)
#         weights = self.fc(pooled).view(B, C, 1)
#         return x * weights
    
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
        # Independent dynamic convolution for each channel
        self.independent_conv1 = DynamicConv1d(1, 32, kernel_size, num_groups, bias)
        self.independent_conv2 = DynamicConv1d(1, 32, kernel_size, num_groups, bias)

        # 联合动态卷积
        self.joint_conv = DynamicConv1d(2, 64, kernel_size, num_groups, bias)
        self.mse_block = MSEBlock(64, reduction)
        # 用于融合的可学习权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 独立特征的权重
        self.beta = nn.Parameter(torch.tensor(0.5))   # Weight for joint features

        # Final normalization and activation
        self.norm = nn.BatchNorm1d(64)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, L = x.shape
        assert C == 2, "Input must have 2 channels (PPG and ECG)."

        # Independent dynamic convolutions for each channel
        x1 = self.independent_conv1(x[:, 0:1, :])  # Shape: (B, 16, L)
        x2 = self.independent_conv2(x[:, 1:2, :])  # Shape: (B, 16, L)

        # Concatenate independent features
        independent_features = torch.cat([x1, x2], dim=1)  # Shape: (B, 32, L)

        # Joint dynamic convolution
        joint_features = self.joint_conv(x)  # Shape: (B, 32, L)
        joint_features = self.mse_block(joint_features)  # Apply MSE module

        # Normalize fusion weights
        alpha_normalized = torch.sigmoid(self.alpha)  # Scale to [0, 1]
        beta_normalized = torch.sigmoid(self.beta)    # Scale to [0, 1]

        # Fuse independent and joint features
        fused_features = alpha_normalized * independent_features + beta_normalized * joint_features

        # 应用标准化和激活
        fused_features = self.norm(fused_features)
        fused_features = self.activation(fused_features)

        return fused_features

# DenseNet
class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, middle_channels=128, out_channels=32):
        super(DenseLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels, middle_channels, 1),
            torch.nn.BatchNorm1d(middle_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(middle_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)

class DenseBlock(torch.nn.Sequential):
    def __init__(self, layer_num, growth_rate, in_channels, middele_channels=128):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels + i * growth_rate, middele_channels, growth_rate)
            self.add_module('denselayer%d' % (i), layer)

class Transition(torch.nn.Sequential):
    def __init__(self, channels):
        super(Transition, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm1d(channels))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv1d(channels, channels // 2, 3, padding=1))
        self.add_module('Avgpool', torch.nn.AvgPool1d(2))

class DenseNet(torch.nn.Module):
    def __init__(self, layer_num=(6, 12, 24, 16), growth_rate=32, init_features=64, in_channels=2, middele_channels=128, output_dim=1):
        super(DenseNet, self).__init__()
        self.feature_channel_num = init_features
        self.conv = torch.nn.Conv1d(in_channels, self.feature_channel_num, 7, 2, 3)
        self.norm = torch.nn.BatchNorm1d(self.feature_channel_num)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(3, 2, 1)

        self.DenseBlock1 = DenseBlock(layer_num[0], growth_rate, self.feature_channel_num, middele_channels)
        self.feature_channel_num = self.feature_channel_num + layer_num[0] * growth_rate
        self.Transition1 = Transition(self.feature_channel_num)

        self.DenseBlock2 = DenseBlock(layer_num[1], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[1] * growth_rate
        self.Transition2 = Transition(self.feature_channel_num)

        self.DenseBlock3 = DenseBlock(layer_num[2], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[2] * growth_rate
        self.Transition3 = Transition(self.feature_channel_num)

        self.DenseBlock4 = DenseBlock(layer_num[3], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[3] * growth_rate

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.feature_channel_num, self.feature_channel_num // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_channel_num // 2, output_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.DenseBlock1(x)
        x = self.Transition1(x)

        x = self.DenseBlock2(x)
        x = self.Transition2(x)

        x = self.DenseBlock3(x)
        x = self.Transition3(x)

        x = self.DenseBlock4(x)
        x = self.avgpool(x)
        # x = x.view(-1, self.feature_channel_num)
        # x = self.regressor(x)
        x = torch.flatten(x, 1)

        return x
    
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
    
class CombinedNet(nn.Module):
    def __init__(self, cornet_output_size=512, cornet_dim=1000, n_cornet_blocks=2, num_BP=1):
        super(CombinedNet, self).__init__()
        
        # 将 ResNet 初始化为特征提取器
        self.resnet = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=2, output_dim=num_BP)
        # 初始化 CorNet 以进行上下文增强
        self.cornet = CorNet(output_size=cornet_output_size, cornet_dim=cornet_dim, n_cornet_blocks=n_cornet_blocks)
        # 添加最终线性层以将 CorNet 输出映射到单个 BP 预测
        self.final_fc = nn.Linear(cornet_output_size, num_BP)

    def forward(self, x):
        # 第 1 步：将输入传递给 ResNet 进行特征提取
        resnet_features = self.resnet.forward(x)  # 应用 ResNet 特征提取器（省略最终 FC 层）
        # 步骤 2：使用 CorNet 增强和融合 ResNet 特征中的上下文
        cornet_output = self.cornet(resnet_features)
        # 步骤 3：将 CorNet 输出映射到最终 BP 预测
        output = self.final_fc(cornet_output)
        
        return output

def DesCor_Net():
    return CombinedNet(cornet_output_size=1024, cornet_dim=512, n_cornet_blocks=2, num_BP=1)