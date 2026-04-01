
# Beyond Johnson-Lindenstrauss: Uniform Bounds for Sketched Bilinear Forms

This implementation is a modification of the  [EE-Net](https://github.com/banyikun/EE-Net-ICLR-2022/tree/master/data) code repository.



## Run:
```

### Synthetic Datasets:

To test for sparsity, we generate synthetic data  where the contexts and / or action are sampled from $l_1$ and $l_2$ ball with sparsity (entries zero'ed out). Look at the `data/generate_data.py` file for $l_2$ ball and `data/l1_ball.py` for $l_1$ ball, where `--context_sparsity` refers to the fraction of zero'ed out elements the context and `--action_sparsity` refers to the same for action.

To generate :
```
cd data
python l2_ball.py --d 200 --K 4 --T 1000 --action_sparsity 0.9 --context_sparsity 0.9
```

To run on synthetic datasets change the dataloader in `baselines/baselines_run.py`:

```
b = load_synthetic_dataloader(data_method,d,K,T,context_sparsity,action_sparsity)
```
Example run:

```
cd baselines
python baselines_run.py --method SketchLinUCB --b_sketch 10 --d 200 --K 4 --T 1000 --csp 0.9 --asp 0.9 --data_method l2_ball 
```

### Flags:

1. `--csp` : context sparsity in the generated data
2. `--asp` : action sparsity in the generated data
3. `--data_method`: `l1_ball` / `l2_ball`

## Prerequisites: 

python 3.8.8, CUDA 11.2, torch 1.9.0, torchvision 0.10.0, sklearn 0.24.1, numpy 1.20.1, scipy 1.6.2, pandas 1.2.4










