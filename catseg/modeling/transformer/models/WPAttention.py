import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from einops.layers.torch import Rearrange
# from pvtv2 import pvt_v2_b2,pvt_v2_b4
from einops import rearrange, repeat

from typing import Optional
from torch import nn, Tensor
import math, copy
from position_encoding import PositionEmbeddingSine
from transformer import Transformer, SelfAttentionLayer, FFNLayer, MLP, _get_activation_fn, _get_clones

import wavelet

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class MSCW(nn.Module):
    def __init__(self, d_model=64):
        super(MSCW, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
        )
        self.local_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.global_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.linear = nn.Linear(d_model,d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pool = torch.mean(x, dim=1, keepdim=True)
        attn = self.local_attn(x) + self.global_attn(pool)
        attn = self.sigmoid(attn)
        return attn
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
        )

    def forward(self, x):
        return self.conv(x)
    
class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        
        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        # print(Q.shape)
        # print(K.shape)
        # print(values.shape)
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.v1 = nn.Linear(hidden_dim, guidance_dim)
        if attention_type == 'linear':
            self.attention = LinearAttention()
        elif attention_type == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError
    
    def forward(self, x, guidance):
        """
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        q = self.q(guidance)
        k = self.k(x)
        v = self.v(x)

        q = rearrange(q, 'B L (H D) -> B L H D', H=self.nheads)
        k = rearrange(k, 'B S (H D) -> B S H D', H=self.nheads)
        v = rearrange(v, 'B S (H D) -> B S H D', H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, 'B L H D -> B L (H D)')
        return out
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, proto_size, guidance_channels, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.norm1 = nn.LayerNorm(d_model)


        self.pool = wavelet.WavePool(d_model)
        self.self_attn1 = nn.MultiheadAttention(d_model, h, dropout=dropout, batch_first=True)
        self.self_attn1 = CrossAttentionLayer(d_model, guidance_channels, nheads=8, attention_type='linear')
        self.mscw1 = MSCW(d_model=d_model)

        self.proto_size = proto_size
        self.conv3x3 = DSConv3x3(d_model, d_model)
        self.Mheads = nn.Linear(d_model, self.proto_size, bias=False)
        self.mscw2 = MSCW(d_model=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.Mheads1 = nn.Linear(guidance_channels, d_model, bias=False)

    def forward(self, query, key, value, attn_mask=None):

        # query = query.transpose(0, 1)
        # key = key.transpose(0, 1)
        # value = value.transpose(0, 1)
        b, n1, c = value.size()
        hw = int(math.sqrt(n1))
        query = self.Mheads1(query)
        feat = key.transpose(1, 2).view(b, c, hw, hw)
        #
        # print("feat.shape:", feat.shape)
        LL, HL, LH, HH = self.pool(feat)
        high_fre = HL + LH + HH
        # print("high_fre.shape:", high_fre.shape)
        low_fre = LL
        # print("low_fre.shape:", low_fre.shape)
        high_fre = high_fre.flatten(2).transpose(1, 2)
        low_fre = low_fre.flatten(2).transpose(1, 2)
        wei = self.mscw1(high_fre+low_fre)

        fre = wei*high_fre + low_fre
        query1 = query
        x1 = self.self_attn1(fre, query1)
        x = x1
        
#         x1 = self.norm1(x1+query1)

#         # channel attention
#         feat = self.conv3x3(feat).flatten(2).transpose(1, 2)
#         multi_heads_weights = self.Mheads(feat)
#         multi_heads_weights = multi_heads_weights.view((b, n1, self.proto_size))
#         multi_heads_weights = F.softmax(multi_heads_weights, dim=1)
#         protos = multi_heads_weights.transpose(-1, -2) @ key
#         query2 = query
#         print(protos.shape)
#         print(query2.shape)
#         attn = self.mscw2(protos+query2)
#         x2 = query2 * attn + query2
#         x2 = self.norm2(x2)

#         x = x1+x2


        return x

def test_multihead_attention():
    print("开始测试 MultiheadAttention 模块...")
    
    # 1. 初始化参数
    batch_size = 2
    seq_len = 576 # 必须是平方数，因为代码中有 math.sqrt(n1)
    d_model = 256
    num_heads = 8
    
    # 2. 实例化模型
    # 注意：确保上面的 DSConv3x3 和 MultiheadAttention 类已在此作用域内
    model = MultiheadAttention(d_model=d_model, h=num_heads)
    model.eval() # 切换到推理模式
    
    # 3. 构造输入数据
    # 根据代码中的 transpose(0, 1)，输入的 shape 应该是 (Seq_len, Batch, Dim)
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    print(f"输入 Query shape: {query.shape}")
    
    # 4. 前向传播
    try:
        with torch.no_grad():
            output = model(query, key, value)
        
        print("-" * 30)
        print("前向传播成功！")
        print(f"输出 Shape: {output.shape}") # 预期应为 (Seq_len, Batch, Dim)
        
        # 验证输出维度
        assert output.shape == query.shape, "输出维度与输入维度不匹配！"
        print("维度验证通过。")
        
    except Exception as e:
        print(f"测试失败，错误信息: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multihead_attention()