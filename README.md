# Distributionally Robust Bayesian Network Skeleton Learning

This is the official implementation of the following paper accepted to *NeurIPS 2023* (**spotlight**):

> **Distributionally Robust Skeleton Learning of Discrete Bayesian Networks**
> 
> Yeshu Li and Brian D. Ziebart
> 
> *37th Conference on Neural Information Processing Systems (NeurIPS 2023)*
> 
> [[Proceeding]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c80addda8bcd95339921cba7581ac7bd-Abstract-Conference.html) [[Virtual]](https://nips.cc/virtual/2023/poster/71840) [[OpenReview]](https://openreview.net/forum?id=NpyZkaEEun) [[arXiv]](https://arxiv.org/abs/2311.06117) [[SlidesLive]](https://slideslive.com/39009235)

## Requirements

- torch
- numpy
- scipy
- pandas
- pgmpy
- pyCausalFS
- causal-learn
- bnlearn

## Data Preparation

Download the data from [bnlearn](https://www.bnlearn.com/bnrepository/), [BN Repository](https://www.cs.huji.ac.il/w~galel/Repository/) and [Malone et al.](http://bnportfolio.cs.helsinki.fi/) or refer to the [released data](https://github.com/DanielLeee/drslbn/releases/download/pre/data.zip) for the complete data adopted throughout our experiments.

## Usage

### Single run
```shell
python main.py --dataset data/cancer.bif --samples 1000 --noise noisefree --pnoise 0 --method dro_wass --epsilon 1 --threshold 0.1
```

### Experiments

Call `exp_mode_bif()` or `exp_mode_real()` in `main.py` for benchmark data or real-world data respectively.

### Function Call

```Python
import drsl
import util

# data: <class 'pandas.core.frame.DataFrame'>
# method_name: 'dro_wass' | 'dro_kl' | 'reg_lr'
# epsilon: algorithm parameter
est_weight_mat = drsl.skeleton_learn(data, method_name, epsilon)

# thr: a chosen threshold to extract edges
est_skeleton = util.skel_by_threshold(est_weight_mat, thr)
```



## Citation

Please cite our work if you find it useful in your research:

```
@inproceedings{
li2023distributionally,
title={Distributionally Robust Skeleton Learning of Discrete Bayesian Networks},
author={Yeshu Li and Brian D Ziebart},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=NpyZkaEEun}
}
```

## Acknowledgement

This project is based upon work supported by the National Science Foundation under Grant No. 1652530.
