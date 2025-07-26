import torch
import torch.nn as nn
from torch import Tensor

class LSTMRegressor1D(nn.Module):
    """
    输入:  x.shape = (B, 2, L)  —— 与你的ResNet保持一致的 [batch, channels, length]
    输出:  (B, 1) —— 单值回归 (SBP)
    结构:  Conv1d 下采样(步长4) -> BiLSTM(2层) -> 时间维均值池化 -> MLP 回归头
    """
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

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 2, L)
        x = self.stem(x)             # (B, C_stem, L')
        x = x.transpose(1, 2)        # (B, L', C_stem) 以 batch_first=True 送入LSTM

        out, _ = self.lstm(x)        # (B, L', H * num_directions)

        # 时间维聚合（也可尝试max/attn pool）
        feat = out.mean(dim=1)       # (B, H * num_directions)

        y = self.head(feat)          # (B, 1)
        return y


def build_lstm_baseline():
    """
    给出一个推荐配置，参数量~2-3M，速度与精度上与1D-ResNet18较接近。
    如需更接近ResNet18_1D的参数量(更大)，可调大 stem_channels / lstm_hidden / lstm_layers。
    """
    return LSTMRegressor1D(
        num_inputs=2,
        num_outputs=1,
        stem_channels=64,  # 48~96 区间可调
        lstm_hidden=256,   # 192~320 区间可调
        lstm_layers=2,     # 2~3 层
        bidirectional=True,
        lstm_dropout=0.1,
        head_hidden=128,
        head_dropout=0.1,
        use_batchnorm_stem=True,
    )


# -------------- 自检 --------------
if __name__ == "__main__":
    B, C, L = 32, 2, 1250
    x = torch.randn(B, C, L)

    model = build_lstm_baseline()
    y = model(x)
    print("Output shape:", y.shape)  # 期望: (32, 1)

    # 打印可训练参数量，便于与ResNet18_1D对齐
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", f"{n_params/1e6:.2f}M")