#!/bin/bash
# en -> de
if [ -z "$1" ] || [ "$1" = "en-de" ]; then
  mkdir -p models/en-de
  python -m src.trainer \
    --gpu 0 \
    --label-smoothing 0.1 \
    --lang-src en \
    --lang-tgt de \
    --save-path models/en-de \
    --max-epochs 60 \
    --learning-rate 5e-4 \
    --optim adamw \
    --betas 0.9 0.98 \
    --decay-method inverse_sqrt \
    --weight-decay 0.0001 \
    --min-lr 1e-9 \
    --batch-size 800 \
    --max-tokens 4096 \
    --warmup-steps 10000 \
    --gradient-accumulation 2 \
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
fi

# de -> en
if [ -z "$1" ] || [ "$1" = "de-en" ]; then
  mkdir -p models/de-en
  python -m src.trainer \
    --gpu 0 \
    --label-smoothing 0.1 \
    --lang-src de \
    --lang-tgt en \
    --save-path models/de-en \
    --max-epochs 60 \
    --learning-rate 5e-4 \
    --optim adamw \
    --betas 0.9 0.98 \
    --decay-method inverse_sqrt \
    --weight-decay 0.0001 \
    --min-lr 1e-9 \
    --batch-size 800 \
    --max-tokens 4096 \
    --warmup-steps 10000 \
    --gradient-accumulation 2 \
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
fi
