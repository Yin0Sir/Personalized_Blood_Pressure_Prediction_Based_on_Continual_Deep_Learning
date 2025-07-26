import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
def Informer1D():
    return MobileViT1DRegressor(input_dim=2, dim=64, depth=2, patch_size=25, mlp_dim=128, output_dim=1)

if __name__ == "__main__":
    model = Informer1D()
    inputs = torch.randn(16, 2, 1250)  # [B, C, L]
    output = model(inputs)
    print(output.shape)  # [16, 1]