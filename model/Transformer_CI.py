import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerEncoderLayer_Roll(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # 前馈网络（FFN）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 或GELU

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # Dropout层（用于残差连接后）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # # 自注意力（Pre-LN结构）
        # src2 = self.norm1(src)
        # src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask)
        # src = src + self.dropout1(src2)
        #
        # # 前馈网络（Pre-LN结构）
        # src2 = self.norm2(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # src = src + self.dropout2(src2)



        # -------------------
        src_roll = torch.roll(src, shifts=1, dims=1)
        src_d = (src + src_roll) / 2.0

        src_d2 = self.norm1(src_d)
        src_d2, _ = self.self_attn(src_d2, src_d2, src_d2, attn_mask=src_mask)
        src_d = src_d + self.dropout1(src_d2)

        src_d2 = self.norm2(src_d)
        src_d2 = self.linear2(self.dropout(self.activation(self.linear1(src_d2))))
        src_d = src_d + self.dropout2(src_d2)

        return src_d

        # return src

        # src_roll = torch.roll(src, shifts=1, dims=1)
        # src_d = (src + src_roll) / 2.0
        # # 自注意力（Pre-LN结构）
        # src2 = self.norm1(src)
        # src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask)
        # src = src + self.dropout1(src2)
        #
        # # 前馈网络（Pre-LN结构）
        # src2 = self.norm2(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # src = src + self.dropout2(src2)
        #
        #
        #
        # # -------------------
        #
        #
        # src_d2 = self.norm1(src_d)
        # src_d2, _ = self.self_attn(src_d2, src_d2, src_d2, attn_mask=src_mask)
        # src = src + self.dropout1(src_d2)
        #
        # src2 = self.norm2(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # src = src + self.dropout2(src2)
        #
        # return src
class TransformerEncoderLayer_Pad(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1, r=2,s=2):
        super().__init__()
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # 前馈网络（FFN）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 或GELU

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # Dropout层（用于残差连接后）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        #self.pool = nn.AvgPool1d(kernel_size=1, stride=1)
        self.r = r
        self.s = s
        #self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
    def forward(self, src, src_mask=None):
        # # 自注意力（Pre-LN结构）
        # src2 = self.norm1(src)
        # src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask)
        # src = src + self.dropout1(src2)
        #
        # # 前馈网络（Pre-LN结构）
        # src2 = self.norm2(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # src = src + self.dropout2(src2)



        # -------------------
        # src_pooled = self.pool(src.permute(0, 2, 1))
        # src_result = src_pooled.permute(0, 2, 1)

        x_reshaped = src.permute(0, 2, 1)  # [128, 1, 1536]

        # 前768个元素直接复制
        first_half = x_reshaped[:, :, :768]  # [128, 1, 768]

        # 后768个元素做2倍下采样
        second_half_pooled = F.avg_pool1d(x_reshaped[:, :, 768:], kernel_size=self.r, stride=self.s)  # [128, 1, 384]

        # 拼接
        result = torch.cat([first_half, second_half_pooled], dim=-1)  # [128, 1, 1152]
        src_result = result.permute(0, 2, 1)




        src_d2 = self.norm1(src_result)
        src_d2, _ = self.self_attn(src_d2, src_d2, src_d2, attn_mask=src_mask)
        src_result = src_result + self.dropout1(src_d2)

        src_d2 = self.norm2(src_result)
        src_d2 = self.linear2(self.dropout(self.activation(self.linear1(src_d2))))
        src_result = src_result + self.dropout2(src_d2)



        return src_result



class TransformerEncoderLayer_CAU(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # 前馈网络（FFN）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 或GELU

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # Dropout层（用于残差连接后）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 自注意力（Pre-LN结构）
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask)
        src = src + self.dropout1(src2)

        # 前馈网络（Pre-LN结构）
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)



        # -------------------
        # src_roll = torch.roll(src, shifts=1, dims=1)
        # src_d = (src + src_roll) / 2.0
        #
        # src_d2 = self.norm1(src_d)
        # src_d2, _ = self.self_attn(src_d2, src_d2, src_d2, attn_mask=src_mask)
        # src_d = src_d + self.dropout1(src_d2)
        #
        # src_d2 = self.norm2(src_d)
        # src_d2 = self.linear2(self.dropout(self.activation(self.linear1(src_d2))))
        # src_d = src_d + self.dropout2(src_d2)
        #
        # return 0.9 * src + 0.1*src_d

        return src


class TransformerEncoder_CAU(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer_CAU(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.layers1 = TransformerEncoderLayer_CAU(d_model, n_heads, dim_feedforward, dropout)
        self.layers2 = TransformerEncoderLayer_Roll(d_model, n_heads, dim_feedforward, dropout)

    def forward(self, src, src_mask=None):
        src1 = self.layers1(src, src_mask)
        src2 = self.layers2(src1, src_mask)
        out = 0.5*src1 + 0.5*src2
        # for layer in self.layers:
        #     src = layer(src, src_mask)
        return out

class TransformerEncoder_MHead(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer_CAU(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.layers1 = TransformerEncoderLayer_CAU(d_model, n_heads, dim_feedforward, dropout)
        self.layers2 = TransformerEncoderLayer_Pad(d_model, n_heads, dim_feedforward, dropout, 4, 2)

        self.layers3 = TransformerEncoderLayer_Pad(d_model, n_heads, dim_feedforward, dropout, 4, 2)
        self.layers4 = TransformerEncoderLayer_Pad(d_model, n_heads, dim_feedforward, dropout, 4, 2)
        self.layers5 = TransformerEncoderLayer_Pad(d_model, n_heads, dim_feedforward, dropout, 4, 2)
        #self.layers4 = TransformerEncoderLayer_Pad3(d_model, n_heads, dim_feedforward, dropout)

        self.z_len = 768
        self.out_linear = nn.Linear(self.z_len * 2 + int(self.z_len), self.z_len * 2 )

    def forward(self, src, src_mask=None):
        src1 = self.layers1(src, src_mask)
        src2 = self.layers2(src, src_mask)
        # src3 = self.layers3(src2, src_mask)
        # src4 = self.layers4(src3, src_mask)
        #src5 = self.layers4(src4, src_mask)

        out = torch.cat([src1, src2], dim=1)
        #out = src1


        # for layer in self.layers:
        #     src = layer(src, src_mask)
        return out
class TransformerEncoder_CI(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer_CAU(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.layers1 = TransformerEncoderLayer_CAU(d_model, n_heads, dim_feedforward, dropout)
        self.layers2 = TransformerEncoderLayer_Pad(d_model, n_heads, dim_feedforward, dropout, 4, 2)

        self.layers3 = TransformerEncoderLayer_Pad(d_model, n_heads, dim_feedforward, dropout, 4, 2)
        self.layers4 = TransformerEncoderLayer_Pad(d_model, n_heads, dim_feedforward, dropout, 4, 2)
        self.layers5 = TransformerEncoderLayer_Pad(d_model, n_heads, dim_feedforward, dropout, 4, 2)
        #self.layers4 = TransformerEncoderLayer_Pad3(d_model, n_heads, dim_feedforward, dropout)

        self.z_len = 768
        self.out_linear = nn.Linear(self.z_len * 2 + int(self.z_len), self.z_len * 2 )

    def forward(self, src, src_mask=None):
        src1 = self.layers1(src, src_mask)
        src2 = self.layers2(src, src_mask)
        # src3 = self.layers3(src2, src_mask)
        # src4 = self.layers4(src3, src_mask)
        #src5 = self.layers4(src4, src_mask)

        out = torch.cat([src1, src2], dim=1)
        #out = src1


        # for layer in self.layers:
        #     src = layer(src, src_mask)
        return out
