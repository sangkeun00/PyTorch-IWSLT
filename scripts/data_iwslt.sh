#!/bin/bash
if ! [ -x "$(command -v subword-nmt)" ]; then
  echo 'Error: subword-nmt is not installed. Perhaps conda environment is not activate?' >&2
  exit 1
fi
if [ ! -f ./scripts/gdrive.sh ]; then
  wget https://raw.githubusercontent.com/GitHub30/gdrive.sh/master/gdrive.sh -O ./scripts/gdrive.sh
  chmod +x ./scripts/gdrive.sh
fi
mkdir -p data
if [ ! -f ./data/iwslt-2014/test.en ]; then
  cd data
  # ../scripts/gdrive.sh https://drive.google.com/drive/folders/1TbbdOd_mbmVmZjHAvq9XyPSWWhHWVp0M
  ../scripts/gdrive.sh https://drive.google.com/drive/folders/1dlzcAd7u2GQg3uEPmF6vCdSyK3vSfWXT
  # back to root
  cd ..
fi
ROOT_DIR=./data/iwslt-2014
BPE_CODE=$ROOT_DIR/bpe.code
CAT=$ROOT_DIR/train.cat
if [ ! -f $BPE_CODE ]; then
  cat $ROOT_DIR/train.en $ROOT_DIR/train.de > $CAT
  echo learning bpe for $CAT
  subword-nmt learn-bpe -s 10000 < $CAT > $BPE_CODE
fi
if [ ! -f $ROOT_DIR/test.bpe.en ]; then
  for lang in en de; do
    for split in train dev test;
    do
      SRC=$ROOT_DIR/$split.$lang
      TGT=$ROOT_DIR/$split.bpe.$lang
      echo apply bpe for $SRC
      subword-nmt apply-bpe -c $BPE_CODE < $SRC > $TGT
    done
  done
  echo
fi
