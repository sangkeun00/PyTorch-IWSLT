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

## Training
```
python -m src.trainer --gpu 0 --enc-layernorm-before --dec-layernorm-before --label-smoothing 0.1
```

For the learning purpose, we also implemented our trainer with `pytorch-lightning`.
If you want to test this, please replace `src.trainer` with `src.trainer_pl` of the above script. 

With Volta GPU (e.g., 2080Ti), you can further speed up training speed by adding `--fp16` option.

You could also directly execute the script
```
./scripts/train_iwslt.sh
```

## Testing

Use test mode to generate target sentences based on the trained model.
```
python -m src.trainer \
    --mode test \
    --gpu 0 \
    --init-checkpoint [model_path] \
    --decode-method beam \
    --beam-size 5 \
    --output-path [decode_output]
```

Or you could also directly execute the script
```
./scripts/test_iwslt.sh
```
