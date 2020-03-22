#!/bin/bash
# en -> de
mkdir -p outputs
if [ -z "$1" ] || [ "$1" = "en-de" ]; then
  python -m src.bin.avg_models \
    models/en-de/model60.pth \
    models/en-de/model59.pth \
    models/en-de/model58.pth \
    models/en-de/model57.pth \
    models/en-de/model56.pth \
    models/en-de/model55.pth \
    models/en-de/model54.pth \
    models/en-de/model53.pth \
    models/en-de/model52.pth \
    models/en-de/model51.pth \
    --output models/en-de/model.avg.pth

  python -m src.trainer \
    --mode test \
    --init-checkpoint models/en-de/model.avg.pth \
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
    --eval-batch-size 10 \
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
fi
if [ -f outputs/test.en-de ]; then
echo evaluate en-de
fairseq-score -s outputs/test.en-de -r data/iwslt-2014/test.de
fi

if [ -z "$1" ] || [ "$1" = "de-en" ]; then
  python -m src.bin.avg_models \
    models/de-en/model60.pth \
    models/de-en/model59.pth \
    models/de-en/model58.pth \
    models/de-en/model57.pth \
    models/de-en/model56.pth \
    models/de-en/model55.pth \
    models/de-en/model54.pth \
    models/de-en/model53.pth \
    models/de-en/model52.pth \
    models/de-en/model51.pth \
    --output models/de-en/model.avg.pth

  python -m src.trainer \
    --mode test \
    --init-checkpoint models/de-en/model.avg.pth \
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
    --eval-batch-size 10 \
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
fi
if [ -f outputs/test.de-en ]; then
echo evaluate de-en
fairseq-score -s outputs/test.de-en -r data/iwslt-2014/test.en
fi
