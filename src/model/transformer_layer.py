import torch
import torch.nn as nn
import torch.nn.functional as F

from .multihead_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        num_heads,
        attn_dropout=0.,
        act_dropout=0.,
        dropout=0.,
        layernorm_before=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.layernorm_before = layernorm_before
        self.act_dropout = act_dropout
        self.dropout = dropout

        # self-attention part
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            mode='fc',
        )
        self.attn_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)

        # point-wise ffn
        self.ffn1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, embed_dim)
        self.ffn_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ffn1.weight)
        nn.init.xavier_uniform_(self.ffn2.weight)

        if self.ffn1.bias is not None:
            nn.init.constant_(self.ffn1.bias, 0.0)
            nn.init.constant_(self.ffn2.bias, 0.0)

    def forward(self, x, mask):
        # self-attn part
        identity = x
        if self.layernorm_before:
            x = self.attn_layernorm(x)
        x = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.attn_layernorm(x)

        # point-wise ffn part
        identity = x
        if self.layernorm_before:
            x = self.ffn_layernorm(x)
        x = F.relu(self.ffn1(x))
        x = F.dropout(x, p=self.act_dropout, training=self.training)
        x = self.ffn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.ffn_layernorm(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        num_heads,
        attn_dropout=0.,
        act_dropout=0.,
        dropout=0.,
        layernorm_before=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.layernorm_before = layernorm_before
        self.act_dropout = act_dropout
        self.dropout = dropout

        # self-attention part
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            mode='fc',
        )
        self.attn_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)

        # end-dec attention part
        self.enc_dec_attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            mode='fc',
        )
        self.enc_dec_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)

        # point-wise ffn
        self.ffn1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, embed_dim)
        self.ffn_layernorm = nn.LayerNorm(embed_dim, eps=1e-5)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ffn1.weight)
        nn.init.xavier_uniform_(self.ffn2.weight)

        if self.ffn1.bias is not None:
            nn.init.constant_(self.ffn1.bias, 0.0)
            nn.init.constant_(self.ffn2.bias, 0.0)

    def forward(self, x, encoder_out, self_mask, encoder_mask):
        # self-attn part
        identity = x
        if self.layernorm_before:
            x = self.attn_layernorm(x)
        x = self.self_attn(query=x, key=x, value=x, mask=self_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.attn_layernorm(x)

        # enc-dec attention part
        identity = x
        if self.layernorm_before:
            x = self.enc_dec_layernorm(x)
        x = self.enc_dec_attention(
            query=x,
            key=encoder_out,
            value=encoder_out,
            mask=encoder_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.enc_dec_layernorm(x)

        # point-wise ffn part
        identity = x
        if self.layernorm_before:
            x = self.ffn_layernorm(x)
        x = F.relu(self.ffn1(x))
        x = F.dropout(x, p=self.act_dropout, training=self.training)
        x = self.ffn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        if not self.layernorm_before:
            x = self.ffn_layernorm(x)

        return x
