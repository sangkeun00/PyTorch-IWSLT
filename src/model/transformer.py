import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_layer import EncoderLayer, DecoderLayer
from .utils import create_mask, positional_embedding

class Encoder(nn.Module):
    def __init__(
        self,
        enc_embed_dim,
        enc_ffn_dim,
        enc_num_heads,
        enc_num_layers,
        src_dict,
        enc_layernorm_before=False,
        attn_dropout=0.,
        act_dropout=0.,
        embed_dropout=0.,
        dropout=0.,
    ):
        super().__init__()

        self.embed_dim = enc_embed_dim

        # Embedding
        self.embed_scale = enc_embed_dim ** 0.5
        self.embedding = nn.Embedding(len(src_dict), enc_embed_dim)
        self.embed_dropout = embed_dropout

        self.layers = nn.ModuleList()
        for _ in range(enc_num_layers):
            layer = EncoderLayer(
                embed_dim=enc_embed_dim,
                ffn_dim=enc_ffn_dim,
                num_heads=enc_num_heads,
                attn_dropout=attn_dropout,
                act_dropout=act_dropout,
                dropout=dropout,
                layernorm_before=enc_layernorm_before,
            )
            self.layer.append(layer)

        self.last_layernorm = nn.LayerNorm(enc_embed_dim, eps=1e-5)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight,
                        mean=0, std=self.embed_dim ** -0.5)

    def forward(self, src_tokens, src_lengths):
        x = self.embedding(src_tokens) * self.embed_scale
        x = x + positional_embedding(src_tokens)
        x = F.dropout(x, p=self.embed_dropout, training=self.training)

        mask = create_mask(src_lengths)

        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.last_layernorm(x)

        return x
