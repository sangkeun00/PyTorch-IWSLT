#!/bin/bash
# en -> de
mkdir -p outputs
python -m src.trainer \
  --mode test \
  --init-checkpoint models/en-de/model.pth \
  --output-path outputs/test.en-de \
  --gpu -1 \
  --enc-layernorm-before \
  --dec-layernorm-before \
  --lang-src en \
  --lang-tgt de \
  --batch-size 1 \
  --transformer-impl custom \
  --dec-embed-dim 512 \
  --dec-ffn-dim 1024 \
  --dec-num-heads 4 \
  --dec-num-layers 6 \
  --enc-embed-dim 512 \
  --enc-ffn-dim 1024 \
  --enc-num-heads 4 \
  --enc-num-layers 6 \
  --dec-tied-weight True \
  --dropout 0.3 \
  --act-dropout 0.1 \
  --attn-dropout 0.0 \
  --embed-dropout 0.3

python -m src.trainer \
  --mode test \
  --init-checkpoint models/de-en/model.pth \
  --output-path outputs/test.de-en \
  --gpu -1 \
  --enc-layernorm-before \
  --dec-layernorm-before \
  --lang-src de \
  --lang-tgt en \
  --batch-size 1 \
  --transformer-impl custom \
  --dec-embed-dim 512 \
  --dec-ffn-dim 1024 \
  --dec-num-heads 4 \
  --dec-num-layers 6 \
  --enc-embed-dim 512 \
  --enc-ffn-dim 1024 \
  --enc-num-heads 4 \
  --enc-num-layers 6 \
  --dec-tied-weight True \
  --dropout 0.3 \
  --act-dropout 0.1 \
  --attn-dropout 0.0 \
  --embed-dropout 0.3
