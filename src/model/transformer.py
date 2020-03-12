import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_layer import EncoderLayer, DecoderLayer
from .positional_embedding import PositionalEmbedding
from .utils import create_mask


class Transformer(nn.Module):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__()
        self.encoder = TransformerEncoder(
            enc_embed_dim=args.enc_embed_dim,
            enc_ffn_dim=args.enc_ffn_dim,
            enc_num_heads=args.enc_num_heads,
            enc_num_layers=args.enc_num_layers,
            src_dict=src_dict,
            enc_layernorm_before=args.enc_layernorm_before,
            attn_dropout=args.attn_dropout,
            act_dropout=args.act_dropout,
            embed_dropout=args.embed_dropout,
            dropout=args.dropout)

        self.decoder = TransformerDecoder(
            dec_embed_dim=args.dec_embed_dim,
            dec_ffn_dim=args.dec_ffn_dim,
            dec_num_heads=args.dec_num_heads,
            dec_num_layers=args.dec_num_layers,
            tgt_dict=tgt_dict,
            dec_layernorm_before=args.enc_layernorm_before,
            attn_dropout=args.attn_dropout,
            act_dropout=args.act_dropout,
            embed_dropout=args.embed_dropout,
            dropout=args.dropout)

    def forward(self, src_tokens, src_lengths, tgt_tokens, tgt_lengths):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(encoder_out, src_lengths, tgt_tokens, tgt_lengths)
        return decoder_out

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()



class TransformerEncoder(nn.Module):
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
        self.embed_scale = math.sqrt(enc_embed_dim)
        self.embedding = nn.Embedding(len(src_dict), enc_embed_dim)
        self.embed_dropout = embed_dropout
        self.positional_embedding = PositionalEmbedding(enc_embed_dim)

        # Encoder layers
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
            self.layers.append(layer)

        # Final LayerNorm
        self.last_layernorm = nn.LayerNorm(enc_embed_dim, eps=1e-5)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight,
                        mean=0, std=1/self.embed_scale)

    def forward(self, src_tokens, src_lengths):
        """forward

        :param src_tokens: [B, T]
        :param src_lengths: [B]
        """
        x = self.embedding(src_tokens) * self.embed_scale
        x = x + self.positional_embedding(src_tokens)
        x = F.dropout(x, p=self.embed_dropout, training=self.training)

        mask = create_mask(src_lengths, max_length=src_tokens.size()[-1])

        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.last_layernorm(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dec_embed_dim,
        dec_ffn_dim,
        dec_num_heads,
        dec_num_layers,
        tgt_dict,
        dec_layernorm_before=False,
        attn_dropout=0.,
        act_dropout=0.,
        embed_dropout=0.,
        dropout=0.,
    ):
        super().__init__()

        self.embed_dim = dec_embed_dim

        # Embedding
        self.embed_scale = math.sqrt(dec_embed_dim)
        self.embedding = nn.Embedding(len(tgt_dict), dec_embed_dim)
        self.embed_dropout = embed_dropout
        self.positional_embedding = PositionalEmbedding(dec_embed_dim)

        # Decoder Layers
        self.layers = nn.ModuleList()
        for _ in range(dec_num_layers):
            layer = DecoderLayer(
                embed_dim=dec_embed_dim,
                ffn_dim=dec_ffn_dim,
                num_heads=dec_num_heads,
                attn_dropout=attn_dropout,
                act_dropout=act_dropout,
                dropout=dropout,
                layernorm_before=dec_layernorm_before,
            )
            self.layers.append(layer)

        # Final LayerNorm
        self.last_layernorm = nn.LayerNorm(dec_embed_dim, eps=1e-5)
        self.out_linear = nn.Linear(dec_embed_dim, len(tgt_dict))

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight,
                        mean=0, std=1/self.embed_scale)

    def forward(self, encoder_out, src_lengths, tgt_tokens, tgt_lengths, cache=None):
        x = self.embedding(tgt_tokens) * self.embed_scale
        x = x + self.positional_embedding(tgt_tokens)
        x = F.dropout(x, p=self.embed_dropout, training=self.training)

        # TODO: add src_mask
        # src_mask =
        mask = create_mask(
                tgt_lengths,
                max_length=tgt_tokens.size()[-1],
                causal=True)
        encoder_mask = create_mask(
                src_lengths,
                max_length=encoder_out.size()[0])
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x,
                    encoder_out=encoder_out,
                    self_mask=mask,
                    encoder_mask=encoder_mask)

        x = self.last_layernorm(x)
        x = self.out_linear(x)

        return x
