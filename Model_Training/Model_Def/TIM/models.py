import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, layer_num=(6, 12, 24, 16), growth_rate=32, init_features=64, in_channels=1, middele_channels=128, output_dim=1):
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

        # 修改为回归输出
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
        x = x.view(-1, self.feature_channel_num)
        x = self.regressor(x)

        return x

# EfficientNetBn
class SEModule(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(SEModule, self).__init__()
        self.avepool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(in_channel, in_channel // ratio)
        self.linear2 = nn.Linear(in_channel // ratio, in_channel)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        b, c, _ = input.shape
        x = self.avepool(input)
        x = x.view([b, c])
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.hardsigmoid(x)
        x = x.view([b, c, 1])
        return input * x

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=4):
        super(MBConvBlock, self).__init__()
        self.out_channels = out_channels  # 添加这一行
        expanded_channels = int(in_channels * expand_ratio)
        self.expand_conv = nn.Conv1d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(expanded_channels)
        self.depthwise_conv = nn.Conv1d(
            expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, groups=expanded_channels, bias=False
        )
        self.bn2 = nn.BatchNorm1d(expanded_channels)
        self.se = SEModule(expanded_channels, se_ratio)
        self.linear_bottleneck = nn.Conv1d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.use_skip_connection = (stride == 1) and (in_channels == out_channels)
        self.leakyrelu = nn.LeakyReLU(0.02)

    def forward(self, x):
        identity = x
        x = self.leakyrelu(self.bn1(self.expand_conv(x)))
        x = self.leakyrelu(self.bn2(self.depthwise_conv(x)))
        x = self.se(x)
        x = self.bn3(self.linear_bottleneck(x))
        if self.use_skip_connection:
            x = identity + x
        return x

class EfficientNet(nn.Module):
    def __init__(self, width_coefficient=1.0, depth_coefficient=1.1, dropout_rate=0.2, in_channels=2, output_dim=1):
        super(EfficientNet, self).__init__()
        # Stem
        out_channels = int(32 * width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.02)
        )

        # Blocks Configuration
        self.blocks_config = [
            (1, 16, 1, 3, 1),
            (6, 24, 2, 3, 2),
            (6, 40, 2, 5, 2),
            (6, 80, 3, 3, 2),
            (6, 112, 3, 5, 1),
            (6, 192, 4, 5, 2),
            (6, 320, 1, 3, 1),
        ]

        # Blocks
        self.blocks = self._make_blocks(out_channels, width_coefficient, depth_coefficient)

        # Head
        head_channels = int(1280 * width_coefficient)
        self.head = nn.Sequential(
            nn.Conv1d(self.out_channels, head_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(head_channels),
            nn.LeakyReLU(0.02)
        )

        # Global Pooling and Classifier
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(head_channels, output_dim)

    def _make_blocks(self, input_channels, width_coefficient, depth_coefficient):
        blocks = []
        for expand_ratio, channels, repeats, kernel_size, stride in self.blocks_config:
            output_channels = int(channels * width_coefficient)
            num_repeats = int(repeats * depth_coefficient)

            # First block in each stage
            blocks.append(MBConvBlock(input_channels, output_channels, expand_ratio, kernel_size, stride))
            input_channels = output_channels

            # Remaining blocks in the stage
            for _ in range(1, num_repeats):
                blocks.append(MBConvBlock(input_channels, output_channels, expand_ratio, kernel_size, stride=1))

        self.out_channels = input_channels  # 更新最后一个块的输出通道数
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def start_flops_count(self):
        pass
    def stop_flops_count(self):
        pass
    def reset_flops_count(self):
        pass
    def compute_average_flops_cost(self):
        pass 

# VGG
class VGG(nn.Module):
    def __init__(self, config, in_channels=1, output_dim=1):
        super(VGG, self).__init__()
        self.feature = self._make_layers(config, in_channels)
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7, 1024),  # 512 是最后一个卷积层的通道数，7 是 AdaptiveAvgPool 的输出大小
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),  # 修改为回归输出
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)  # 展平成 (batch_size, 512*7)
        x = self.regressor(x)
        return x

    @staticmethod
    def _make_layers(config, in_channels):
        layers = []
        for v in config:
            if v == "M":  # 表示 MaxPool 层
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv1d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm1d(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        layers += [nn.AdaptiveAvgPool1d(7)]  # 最后加入 AdaptiveAvgPool
        return nn.Sequential(*layers)

# 定义 VGG 的配置
VGG_CONFIGS = {
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

# Xception
class SeparableConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SeparableConv1d, self).__init__()

        # 深度卷积
        self.depthwise = torch.nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        # 逐点卷积
        self.pointwise = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Entry(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.beforeresidual = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 32, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 3, 2, 1),
            torch.nn.ReLU()
        )

        self.residual_branch1 = torch.nn.Conv1d(64, 128, 1, 2)
        self.residual_model1 = torch.nn.Sequential(
            SeparableConv1d(64, 128, 3, 1, 1),
            torch.nn.ReLU(),
            SeparableConv1d(128, 128, 3, 1, 1),
            torch.nn.MaxPool1d(3, 2, 1)
        )

        self.residual_branch2 = torch.nn.Conv1d(256, 256, 1, 2)
        self.residual_model2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(256, 256, 3, 1, 1),
            torch.nn.ReLU(),
            SeparableConv1d(256, 256, 3, 1, 1),
            torch.nn.MaxPool1d(3, 2, 1)
        )

        self.residual_branch3 = torch.nn.Conv1d(512, 728, 1, 2)
        self.residual_model3 = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(512, 728, 3, 1, 1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
            torch.nn.MaxPool1d(3, 2, 1)
        )

    def forward(self, x):
        x = self.beforeresidual(x)

        x1 = self.residual_branch1(x)
        x = self.residual_model1(x)
        x = torch.cat([x, x1], dim=1)

        x1 = self.residual_branch2(x)
        x = self.residual_model2(x)
        x = torch.cat([x, x1], dim=1)

        x1 = self.residual_branch3(x)
        x = self.residual_model3(x)
        x = x + x1

        return x

class Middleflow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.layers(x)

class Exitflow(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.residual = torch.nn.Conv1d(728, 1024, 1, 2)
        self.residual_model = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 1024, 3, 1, 1),
            torch.nn.MaxPool1d(3, 2, 1)
        )
        self.last_layer = torch.nn.Sequential(
            SeparableConv1d(1024, 1536, 3, 1, 1),
            torch.nn.ReLU(),
            SeparableConv1d(1536, 2048, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, output_dim)  # 修改为回归任务的输出
        )

    def forward(self, x):
        x = self.residual_model(x) + self.residual(x)
        x = self.last_layer(x)

        return x

class Xception(torch.nn.Module):
    def __init__(self, in_channels, output_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            Entry(in_channels),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Exitflow(output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# MobileNetV3
class conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, keral,stride=1, groups=1,activation = None):
        super().__init__()

        padding = keral//2
        self.use_activation = activation
        self.conv = torch.nn.Conv1d(in_channels, out_channels, keral, stride,padding, groups=groups)
        self.bath = torch.nn.BatchNorm1d(out_channels)
        if self.use_activation == 'Relu':
            self.activation = torch.nn.ReLU6()
        elif self.use_activation == 'H_swish':
            self.activation = torch.nn.Hardswish()

    def forward(self,x):
        x = self.conv(x)
        if x.size()[-1] != 1:
            x = self.bath(x)
        if self.use_activation != None:
            x = self.activation(x)
        return x

class bottleneck(torch.nn.Module):
    def __init__(self,in_channels,keral_size,expansion_size,out_channels,use_attenton = False,activation = 'Relu',stride=1):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attenton = use_attenton

        self.conv = conv(in_channels,expansion_size,1,activation=activation)
        self.conv1 = conv(expansion_size,expansion_size,keral_size,stride=stride,groups=expansion_size,activation=activation)

        if self.use_attenton:
            self.attenton = SE_block(expansion_size)

        self.conv2 = conv(expansion_size,out_channels,1,activation=activation)

    def forward(self,x):

        x1 = self.conv(x)
        x1 = self.conv1(x1)
        if self.use_attenton:
            x1 = self.attenton(x1)
        x1 = self.conv2(x1)

        if self.stride == 1 and self.in_channels == self.out_channels:
            x1 += x

        return x1

class SE_block(torch.nn.Module):
    def __init__(self,in_channel,ratio=1):
        super(SE_block, self).__init__()
        self.avepool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear1 = torch.nn.Linear(in_channel,in_channel//ratio)
        self.linear2 = torch.nn.Linear(in_channel//ratio,in_channel)
        self.Hardsigmoid = torch.nn.Hardsigmoid(inplace=True)
        self.Relu = torch.nn.ReLU(inplace=True)

    def forward(self,input):
        b,c,_ = input.shape
        x = self.avepool(input)
        x = x.view([b,c])
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.Hardsigmoid(x)
        x = x.view([b,c,1])

        return input*x

class MobileNetV3_large(torch.nn.Module):
    def __init__(self, in_channels, output_dim=1):
        super().__init__()
        self.features = torch.nn.Sequential(
            conv(in_channels, 16, 3, 2, activation='H_swish'),
            bottleneck(16, 3, 16, 16, False, 'Relu', 1),
            bottleneck(16, 3, 64, 24, False, 'Relu', 2),
            bottleneck(24, 3, 72, 24, False, 'Relu', 1),
            bottleneck(24, 5, 72, 40, True, 'Relu', 2),
            bottleneck(40, 5, 120, 40, True, 'Relu', 1),
            bottleneck(40, 5, 120, 40, True, 'Relu', 1),
            bottleneck(40, 3, 240, 80, False, 'H_swish', 2),
            bottleneck(80, 3, 200, 80, False, 'H_swish', 1),
            bottleneck(80, 3, 184, 80, False, 'H_swish', 1),
            bottleneck(80, 3, 184, 80, False, 'H_swish', 1),
            bottleneck(80, 3, 480, 112, True, 'H_swish', 1),
            bottleneck(112, 3, 672, 112, True, 'H_swish', 1),
            bottleneck(112, 5, 672, 160, True, 'H_swish', 2),
            bottleneck(160, 5, 960, 160, True, 'H_swish', 2),
            bottleneck(160, 5, 960, 160, True, 'H_swish', 2),
            conv(160, 960, 1, 1, activation='H_swish'),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        # 修改分类器为回归输出层
        self.regressor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(960, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim),  # 输出回归值
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
    
# MobileViT1D
# —— 局部卷积块 + MobileViT 短块组合 —— #
class MobileViTBlock1D(nn.Module):
    def __init__(self, dim, kernel_size=3, patch_size=25, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.norm = nn.BatchNorm1d(dim)

        self.patch_size = patch_size
        self.dim = dim
        flatten_dim = patch_size * dim  # 每个 patch 展平后的维度
        # 相当于 transformer 中的跨 patch attention
        self.transformer = nn.Sequential(
            nn.Linear(flatten_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, flatten_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, C, L]
        y = self.conv1(x)
        y = self.conv2(F.relu(y))
        y = self.norm(y)
        # 切 patch
        B, C, L = y.shape
        y = y.view(B, C, L // self.patch_size, self.patch_size)
        y = y.permute(0, 2, 3, 1).contiguous()  # [B, Npatch, patch_size, C]
        y = y.flatten(2)  # 每 patch 展平 [B, Npatch, patch_size*C]
        y = self.transformer(y)  # mix across patches
        # 重构形状
        y = y.view(B, L // self.patch_size, self.patch_size, C)\
             .permute(0, 3, 1, 2).contiguous()
        y = y.view(B, C, L)
        return F.relu(x + y)

# —— Full MobileViT1D-inspired 回归模型 —— #
class MobileViT1DRegressor(nn.Module):
    def __init__(self, input_dim=2, dim=32, depth=3, patch_size=25, mlp_dim=64, output_dim=1):
        super().__init__()
        self.stem = nn.Conv1d(input_dim, dim, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[
            MobileViTBlock1D(dim, kernel_size=3, patch_size=patch_size, mlp_dim=mlp_dim)
            for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dim, output_dim)

    def forward(self, x):
        # x: [B, 2, L]
        x = self.stem(x)          # [B, dim, L]
        x = self.blocks(x)       # MobileViT 模块
        x = self.pool(x).squeeze(-1)  # [B, dim]
        return self.fc(x)        # [B, output_dim]
    
# LSTM hard 回归器
class LSTMRegressor1D(nn.Module):
    def __init__(
        self,
        num_inputs: int = 2,         # 输入通道数（你的任务是2通道）
        num_outputs: int = 1,        # 回归输出维度（SBP=1）
        stem_channels: int = 64,     # 卷积stem输出通道数(影响参数量与效果)
        lstm_hidden: int = 256,      # LSTM隐层维度(影响参数量与效果)
        lstm_layers: int = 2,        # LSTM层数(>=2时才会用到dropout)
        bidirectional: bool = True,  # 双向LSTM更稳
        lstm_dropout: float = 0.1,   # LSTM层间dropout
        head_hidden: int = 128,      # MLP头的中间维度
        head_dropout: float = 0.1,   # MLP头的dropout
        use_batchnorm_stem: bool = True,  # stem中是否使用BN
    ):
        super().__init__()

        # ---- 1) 卷积 Stem：时域降采样，降低LSTM时序长度 ----
        # 长度 L -> ceil(L/4), 形状: (B, num_inputs, L) -> (B, stem_channels, L/4)
        stem = [
            nn.Conv1d(num_inputs, stem_channels, kernel_size=5, stride=4, padding=2, bias=False),
        ]
        if use_batchnorm_stem:
            stem += [nn.BatchNorm1d(stem_channels)]
        stem += [nn.ReLU(inplace=True)]
        self.stem = nn.Sequential(*stem)

        # ---- 2) BiLSTM 主干 ----
        self.lstm = nn.LSTM(
            input_size=stem_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,          # (B, T, C)
            bidirectional=bidirectional,
            dropout=(lstm_dropout if lstm_layers > 1 else 0.0),
        )

        # ---- 3) 读出头：时间维均值池化 + MLP -> 回归值 ----
        proj_in = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(proj_in, head_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden, num_outputs, bias=True),
        )

        self._init_weights_lstm_forget_bias()

    def _init_weights_lstm_forget_bias(self):
        # 常见的稳定训练初始化：把LSTM的forget gate偏置初始化为正值(如1.0)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
                # PyTorch LSTM门的顺序: [i, f, g, o]
                hidden = param.shape[0] // 4
                param.data[hidden:2*hidden] = 1.0  # forget gate偏置=1

    def forward(self, x):
        # x: (B, 2, L)
        x = self.stem(x)             # (B, C_stem, L')
        x = x.transpose(1, 2)        # (B, L', C_stem) 以 batch_first=True 送入LSTM
        out, _ = self.lstm(x)        # (B, L', H * num_directions)
        # 时间维聚合（也可尝试max/attn pool）
        feat = out.mean(dim=1)       # (B, H * num_directions)
        y = self.head(feat)          # (B, 1)
        return y

def build_lstm_baseline():
    return LSTMRegressor1D(
        num_inputs=2,
        num_outputs=1,
        stem_channels=64,  # 48~96 区间可调
        lstm_hidden=320,   # 192~320 区间可调
        lstm_layers=3,     # 2~3 层
        bidirectional=True,
        lstm_dropout=0.1,
        head_hidden=128,
        head_dropout=0.1,
        use_batchnorm_stem=True,
    )

def lstm_o3_1d():
    return build_lstm_baseline()
    
def MobileViT1D():
    return MobileViT1DRegressor(input_dim=2, dim=64, depth=2, patch_size=25, mlp_dim=128, output_dim=1)
    
def VGG16():
    return VGG(VGG_CONFIGS["VGG16"], in_channels=2, output_dim=1)
def VGG19():
    return VGG(VGG_CONFIGS["VGG19"], in_channels=2, output_dim=1)
def DenseNet_1D():
    return DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=2, output_dim=1)

def Xception_1D():
    return Xception(in_channels=2, output_dim=1)
# def efficientnet_b0():
#     return EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, in_channels=2, output_dim=1)
def efficientnet_b1():
    return EfficientNet(width_coefficient=1.0, depth_coefficient=1.1, dropout_rate=0.2, in_channels=2, output_dim=1)
def efficientnet_b2():
    return EfficientNet(width_coefficient=1.1, depth_coefficient=1.2, dropout_rate=0.3, in_channels=2, output_dim=1)
def efficientnet_b3():
    return EfficientNet(width_coefficient=1.2, depth_coefficient=1.4, dropout_rate=0.3, in_channels=2, output_dim=1)
# def efficientnet_b4():
#     return EfficientNet(width_coefficient=1.4, depth_coefficient=1.8, dropout_rate=0.4, in_channels=2, output_dim=1)
def MobileNetV3():
    return MobileNetV3_large(in_channels=2, output_dim=1)
    
if __name__ == "__main__":
    model = MobileViT1D()
    inputs = torch.randn(32, 2, 1250)  # [B, C, L]
    output = model(inputs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(output.shape)  # [32, 1]
    print("Trainable parameters:", n_params)