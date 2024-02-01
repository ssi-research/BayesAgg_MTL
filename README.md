# BayesAgg-MTL

### Installation Instructions
1. Install repo:
```bash
conda create -n "BayesAgg-MTL" python=3.9
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
```

2. Download the UTKFace dataset from the offical repository [[Link]](https://susanqq.github.io/UTKFace/) or kaggle [[Link]](https://www.kaggle.com/datasets/jangedoo/utkface-new) and place it under experiments/utkface/dataset

### Running th code
```bash
cd experiments/utkface
python trainer.py
```
