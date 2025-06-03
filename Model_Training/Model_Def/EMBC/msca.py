import math
import torch
import torch.nn as nn
import typing as t
"""
多光谱通道注意力(MSCA)
论文题目:FcaNet: Frequency Channel Attention Networks
中文题目:FcaNet: 频域通道注意力网络
论文链接:https://arxiv.org/pdf/2012.11879
官方github:https://github.com/cfzd/FcaNet
所属机构：浙江大学计算机学院，浙江大学上海高等研究院
关键词：频域，通道注意力，图像分类，目标检测，语义分割
"""
def get_freq_indices(method: str) -> t.List[int]:
    """
    Get the frequency indices for 1D signals according to the method.
    """
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        indices = list(range(0, num_freq))  # 高频索引从 0 开始
    elif 'low' in method:
        indices = list(range(-num_freq, 0))  # 低频索引从 -num_freq 开始
    else:
        raise NotImplementedError
    return indices

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate DCT filters for 1D time-series data
    """

    def __init__(self, seq_len: int, freq_indices: t.List[int], channel: int):
        super(MultiSpectralDCTLayer, self).__init__()

        self.num_freq = len(freq_indices)

        # 固定的 DCT 滤波器
        self.register_buffer('weight', self.get_dct_filter(seq_len, freq_indices, channel))

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape [B, C, T]
        Returns:
            Tensor with shape [B, C]
        """
        assert len(x.shape) == 3, 'x must be 3 dimensions, but got ' + str(len(x.shape))

        # 应用 DCT 滤波器
        x = x * self.weight  # [B, C, T]

        # 在时间维度上求和，得到每个通道的频域响应
        result = torch.sum(x, dim=2)  # [B, C]
        return result

    def build_filter(self, pos, freq, POS):
        """
        Construct a DCT basis vector
        """
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, seq_len: int, freq_indices: t.List[int], channel: int):
        """
        Constructing different frequency vectors for 1D signals
        """
        dct_filter = torch.zeros(channel, seq_len)

        c_part = channel // len(freq_indices)

        for i, freq in enumerate(freq_indices):
            for t in range(seq_len):
                dct_filter[i * c_part: (i + 1) * c_part, t] = self.build_filter(t, freq, seq_len)

        return dct_filter
    
class MultiSpectralAttentionLayer(nn.Module):
    """
    Multi-Spectral Attention for 1D Time-Series Data
    """

    def __init__(self, channel: int, seq_len: int, reduction: int = 16, freq_sel_method: str = 'top8'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.seq_len = seq_len

        # 获取选定的频率索引
        freq_indices = get_freq_indices(freq_sel_method)
        self.num_split = len(freq_indices)

        # 生成 1D 的 DCT 滤波器
        self.dct_layer = MultiSpectralDCTLayer(seq_len, freq_indices, channel)

        # 通道注意力部分
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape [B, C, T]
        Returns:
            Output tensor with the same shape as input [B, C, T]
        """
        n, c, t = x.shape

        # 如果输入长度和 DCT 滤波器长度不一致，则进行自适应池化
        if t != self.seq_len:
            x_pooled = torch.nn.functional.adaptive_avg_pool1d(x, self.seq_len)
        else:
            x_pooled = x

        # 计算频域特征
        y = self.dct_layer(x_pooled)  # [B, C]

        # 通道注意力生成
        y = self.fc(y).view(n, c, 1)  # [B, C, 1]

        # 特征加权
        return x * y.expand_as(x)

if __name__ == '__main__':
    # 测试输入
    x = torch.randn(32, 2, 1250)  # 输入为 [B, C, T]
    model = MultiSpectralAttentionLayer(channel=2, seq_len=1250)
    output = model(x)
    print(output.shape)
