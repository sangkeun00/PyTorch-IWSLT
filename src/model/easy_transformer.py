import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_embedding import PositionalEmbedding


class EasyTransformer(nn.Module):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__()

        self.args = args
        self.pad_id = src_dict.PAD_ID

        self.transformer = nn.Transformer(
            d_model=args.enc_embed_dim,
            nhead=args.enc_num_heads,
            num_encoder_layers=args.enc_num_layers,
            num_decoder_layers=args.dec_num_layers,
            dim_feedforward=args.enc_ffn_dim,
            dropout=0.1,
            activation='relu',
        )

        self.embed_scale = math.sqrt(args.enc_embed_dim)
        self.enc_embedding = nn.Embedding(len(src_dict), args.enc_embed_dim)
        self.dec_embedding = nn.Embedding(len(tgt_dict), args.dec_embed_dim)
        self.positional_embedding = PositionalEmbedding(args.enc_embed_dim)

        self.out = nn.Linear(args.dec_embed_dim, len(tgt_dict), bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.enc_embedding.weight,
                        mean=0, std=1/self.embed_scale)
        nn.init.normal_(self.dec_embedding.weight,
                        mean=0, std=1/self.embed_scale)
        nn.init.normal_(self.out.weight,
                        mean=0, std=1/self.embed_scale)

    def forward_embedding(self, src_tokens, tgt_tokens):
        src_mask = (src_tokens == self.pad_id).float().unsqueeze(2)
        tgt_mask = (tgt_tokens == self.pad_id).float().unsqueeze(2)

        src = self.embed_scale * self.enc_embedding(src_tokens)
        src = src + self.positional_embedding(src)
        src = F.dropout(src, p=0.1, training=self.training)
        src = src * src_mask

        tgt = self.embed_scale * self.dec_embedding(tgt_tokens)
        tgt = tgt + self.positional_embedding(tgt)
        tgt = F.dropout(tgt, p=0.1, training=self.training)
        tgt = tgt * tgt_mask

        return src, tgt


    def forward(self, src_tokens, src_lengths, tgt_tokens, tgt_lengths):
        src, tgt = self.forward_embedding(src_tokens, tgt_tokens)

        src_mask = None                                 # Should be None
        tgt_mask = self.generate_tgt_mask(tgt_tokens)   # Need to check
        memory_mask = None                              # Need to check
        src_key_padding_mask = (src_tokens == self.pad_id)     # Need to check
        tgt_key_padding_mask = (tgt_tokens == self.pad_id)     # Need to check
        memory_key_padding_mask = src_key_padding_mask.clone() # Need to check


        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        out = self.transformer.forward(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        out = out.transpose(0, 1)
        out = self.out(out)

        return out

    def generate_tgt_mask(self, tgt_tokens):
        sz = tgt_tokens.shape[1]
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(tgt_tokens.device)
