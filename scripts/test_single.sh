#!/bin/bash
# en -> de
mkdir -p outputs
python -m src.trainer \
  --data-dir test_data/single \
  --mode test \
  --init-checkpoint models/single-en-de/model.pth \
  --output-path outputs/single/test.en-de \
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
