#!/bin/bash
if [ ! -f ./scripts/gdrive.sh ]; then
  wget https://raw.githubusercontent.com/GitHub30/gdrive.sh/master/gdrive.sh -O ./scripts/gdrive.sh
  chmod +x ./scripts/gdrive.sh
fi
mkdir -p data
if [ ! -f ./data/iwslt-2014/test.en ]; then
  cd data
  ../scripts/gdrive.sh https://drive.google.com/drive/folders/1TbbdOd_mbmVmZjHAvq9XyPSWWhHWVp0M
  # back to root
  cd ..
fi
