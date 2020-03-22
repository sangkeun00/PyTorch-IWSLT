# 11747_hw3
Transformer re-implementation from scratch (PyTorch)


## Prepare
To test our model on your machine, please first set up the conda environment with the provided `environment.yml` file, and activate it.

```bash
conda env update -f environment.yml
conda activate nn4nlp-hw3
```

We train/test our model on the IWSLT en-de dataset, as it allows fast experiments due to its small dataset size.
To download and preprocess the IWSLT en-de dataset, please run the following script.

```bash
./scripts/data_iwslt.sh
```

#### Half Precision (fp16) Training
To allow faster training, please install [apex](https://github.com/NVIDIA/apex) from NVIDIA.
The detailed description on its installation is provided in [https://github.com/NVIDIA/apex#quick-start](https://github.com/NVIDIA/apex#quick-start).

In summary, we could use the following commands to install the package:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Training
```bash
python -m src.trainer --gpu 0 --enc-layernorm-before --dec-layernorm-before --label-smoothing 0.1
```

With Volta GPU (e.g., 2080Ti), you can further speed up training speed by adding `--fp16` option.

You could also directly execute the script
```bash
./scripts/train_iwslt.sh
```

## Testing

Use test mode to generate target sentences based on the trained model.
```bash
python -m src.trainer \
    --mode test \
    --gpu 0 \
    --init-checkpoint [model_path] \
    --decode-method beam \
    --beam-size 5 \
    --output-path [decode_output]
```

Once the output is generated, use the following command to remove BPE, and then BLEU can be computed against the ground truth using standard evaluation scripts.

```bash
sed -r 's/(@@ )|(@@ ?$)//g' [decode_output] > [no_bpe_output]
```


Or you could also directly execute the script to automatically run the above steps.
```bash
./scripts/test_iwslt.sh
```
