import torch
from torch import nn
from torch.nn import functional as F

class PatchEmbed(nn.Module):
    """Patch Embedding for 1D time series data."""
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=1, embed_dim=768, norm_layer=dict(type='BN1d'), act_cfg=None):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm1d(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    """Self-Attention for 1D time series."""
    def __init__(self, dim, num_heads=1, qk_scale=None, attn_drop=0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Conv1d(dim, dim, kernel_size=1)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.sr = nn.Conv1d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim) if sr_ratio > 1 else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        k, v = torch.chunk(self.kv(kv), 2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2).reshape(B, C, L)
        return x

class DynamicConv1d(nn.Module):
    """Dynamic Convolution for 1D time series."""
    def __init__(self, dim, kernel_size=3, reduction_ratio=4, num_groups=1, bias=True):
        super().__init__()
        self.num_groups = num_groups
        self.K = kernel_size
        self.weight = nn.Parameter(torch.randn(num_groups, dim, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool1d(kernel_size)
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim // reduction_ratio, kernel_size=1),
            nn.BatchNorm1d(dim // reduction_ratio),
            nn.GELU(),
            nn.Conv1d(dim // reduction_ratio, dim * num_groups, kernel_size=1)
        )
        self.bias = nn.Parameter(torch.randn(num_groups, dim), requires_grad=True) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, L = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = weight.sum(dim=1).reshape(-1, 1, self.K)
        x = F.conv1d(x.reshape(1, -1, L), weight, padding=1, groups=B * C)
        x = x.reshape(B, C, L)
        return x

class HybridTokenMixer(nn.Module):
    """Hybrid Token Mixer for 1D time series."""
    def __init__(self, dim, kernel_size=3, num_groups=2, num_heads=1, sr_ratio=1, reduction_ratio=8):
        super().__init__()
        self.local_unit = DynamicConv1d(dim // 2, kernel_size, num_groups)
        self.global_unit = Attention(dim // 2, num_heads, sr_ratio=sr_ratio)
        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(inner_dim),
            nn.Conv1d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x
        
        return x

class MultiScaleDWConv(nn.Module):
    """Multi-Scale Depthwise Convolution for 1D time series."""
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv1d(channels, channels, kernel_size=scale[i], padding=scale[i] // 2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, self.channels, dim=1)
        out = [conv(feat) for conv, feat in zip(self.proj, x)]
        return torch.cat(out, dim=1)

class Mlp(nn.Module):
    """Multi-Layer Perceptron for 1D time series."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_cfg=dict(type='GELU'), drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(hidden_features)
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.norm = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv1d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_features)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

class LayerScale(nn.Module):
    """Layer Scale for 1D time series."""
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * init_value)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1)

class fusionBlock(nn.Module):
    """Network Block for 1D time series."""
    def __init__(self, dim, kernel_size=3, sr_ratio=1, num_groups=2, num_heads=1, mlp_ratio=4, drop=0, drop_path=0, layer_scale_init_value=1e-5):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pos_embed = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = nn.BatchNorm1d(dim)
        self.token_mixer = HybridTokenMixer(dim, kernel_size, num_groups, num_heads, sr_ratio)
        self.norm2 = nn.BatchNorm1d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = nn.Identity()  # Simplified for demonstration
        self.layer_scale_1 = LayerScale(dim, layer_scale_init_value)
        self.layer_scale_2 = LayerScale(dim, layer_scale_init_value)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.layer_scale_1(self.token_mixer(self.norm1(x))))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))
        return x

class TransXNet_1D(nn.Module):
    """TransXNet for 1D time series regression."""
    def __init__(self, in_chans=2, embed_dim=256, num_blocks=16):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[fusionBlock(embed_dim) for _ in range(num_blocks)])
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 1)  # 输出一个连续值
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.regressor(x)
        return x

def TransXNet():
    return TransXNet_1D(in_chans=2, embed_dim=512, num_blocks=12)
