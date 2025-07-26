import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.3):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 输入为 (batch, seq_len, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=False)

        # 时间维度上做池化后回归
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # for (B, C, L) → (B, C, 1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, 2, L) → (B, L, 2)
        x = x.permute(0, 2, 1)

        # LSTM 输出 (B, L, H)
        out, _ = self.lstm(x)

        # 转为 (B, H, L) → pool → (B, H, 1)
        out = out.permute(0, 2, 1)
        out = self.global_pool(out)

        # 回归 → (B, output_dim)
        out = self.regressor(out)
        return out
    
def lstm1d():
    return LSTMRegressor(input_dim=2, hidden_dim=64, num_layers=2, output_dim=1)

if __name__ == "__main__":
    model = lstm1d()
    inputs = torch.randn(32, 2, 1250)  # [B, C, L]
    output = model(inputs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(output.shape)  # [16, 1]
    print("Trainable parameters:", n_params)