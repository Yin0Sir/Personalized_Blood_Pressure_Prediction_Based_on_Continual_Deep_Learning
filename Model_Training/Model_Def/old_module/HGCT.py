import math
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange

# ----------------- Argument Parsing -----------------
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam optimizer')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    return parser.parse_args()

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# ----------------- Building Blocks -----------------
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens):
        super().__init__()
        self.key_layer = nn.Linear(key_size, num_hiddens, bias=False)
        self.query_layer = nn.Linear(query_size, num_hiddens, bias=False)
        self.value_layer = nn.Linear(num_hiddens, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries, keys):
        # 查询的形状：[bs, query_size]
        # 键的形状：[bs, seq_len, query_size]
        queries = self.query_layer(queries.unsqueeze(1))# [bs, query_size]->[bs, 1, query_size]
        keys = self.key_layer(keys)
        features = torch.tanh(queries + keys)
        scores = self.value_layer(features)
        return self.softmax(scores)

def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            if batch_norm:
                layers.extend([conv, nn.BatchNorm1d(v), nn.ReLU(inplace=True)])
            else:
                layers.append(conv)
            in_channels = v
    return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, use_shortcut=True):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(planes, planes, kernel_size=3, padding=1)
        self.bn1, self.bn2, self.bn3 = [nn.BatchNorm1d(planes) for _ in range(3)]
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = use_shortcut
        self.downsample = (
            nn.Sequential(nn.Conv1d(inplanes, planes, kernel_size=1), nn.BatchNorm1d(planes))
            if inplanes != planes else None
        )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            out += identity
        return self.relu(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + Variable(self.pe[:, :x.size(1)], requires_grad=False))

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.num_heads = num_heads

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            energy.masked_fill_(~mask, float('-inf'))
        attention = F.softmax(energy / math.sqrt(keys.size(-1)), dim=-1)
        attention = self.attention_dropout(attention)
        out = torch.einsum('bhqk, bhkd -> bhqd', attention, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)

class TransformerEncoder(nn.Module):
    def __init__(self, emb_size, num_heads, depth, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, dropout),
                nn.Dropout(dropout),
                nn.LayerNorm(emb_size),
                nn.Linear(emb_size, 4 * emb_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * emb_size, emb_size)
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x

# ----------------- HGCTNet Model -----------------
class HGCTNet(nn.Module):
    def __init__(self, num_classes=2, depth=3, num_heads=4, emb_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            make_layers([32, 32, 'M', 64, 64, 'M'], batch_norm=True),
            BasicBlock(64, 128),
            nn.MaxPool1d(2, stride=3)
        )
        self.attention = AdditiveAttention(111, 47, 8)
        self.transformer = TransformerEncoder(emb_size, num_heads, depth)
        self.regressor = nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, sigs, feas):
        x = self.cnn(sigs)
        alpha = self.attention(feas[:, 6:], x)
        x = torch.mul(alpha, x)
        x = x.mean(dim=-1)
        x = self.transformer(x.unsqueeze(1))
        return self.regressor(x.squeeze(1))
