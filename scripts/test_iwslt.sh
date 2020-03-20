#!/bin/bash
# en -> de
mkdir -p outputs
# if [ ! -f outputs/test.en-de ]; then
  python -m src.trainer \
    --mode test \
    --init-checkpoint models/en-de/model.pth \
    --decode-method beam \
    --beam-size 5 \
    --max-decode-length-multiplier 2.0 \
    --max-decode-length-base 10 \
    --length-normalize True \
    --output-path outputs/test.bpe.en-de \
    --gpu 0 \
    --enc-layernorm-before \
    --dec-layernorm-before \
    --lang-src en \
    --lang-tgt de \
    --batch-size 10 \
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
  sed -r 's/(@@ )|(@@ ?$)//g' outputs/test.bpe.en-de > outputs/test.en-de
# fi
echo evaluate en-de
# ./multi-bleu.perl data/iwslt-2014/test.de < outputs/test.en-de
# sacremoses detokenize -l de < outputs/test.en-de > outputs/test.en-de.detok
# sacremoses detokenize -l de < data/iwslt-2014/test.de > data/iwslt-2014/test.de.detok
sacrebleu --tokenize none data/iwslt-2014/test.de -l en-de < outputs/test.en-de

# if [ ! -f outputs/test.de-en ]; then
  python -m src.trainer \
    --mode test \
    --init-checkpoint models/de-en/model.pth \
    --decode-method beam \
    --beam-size 5 \
    --max-decode-length-multiplier 2.0 \
    --max-decode-length-base 10 \
    --length-normalize True \
    --output-path outputs/test.bpe.de-en \
    --gpu 0 \
    --enc-layernorm-before \
    --dec-layernorm-before \
    --lang-src de \
    --lang-tgt en \
    --batch-size 10 \
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
  sed -r 's/(@@ )|(@@ ?$)//g' outputs/test.bpe.de-en > outputs/test.de-en
# fi
echo evaluate de-en
# ./multi-bleu.perl data/iwslt-2014/test.en < outputs/test.de-en
# sacremoses detokenize -l en < outputs/test.de-en > outputs/test.de-en.detok
# sacremoses detokenize -l en < data/iwslt-2014/test.en > data/iwslt-2014/test.en.detok
sacrebleu --tokenize none data/iwslt-2014/test.en -l de-en < outputs/test.de-en
