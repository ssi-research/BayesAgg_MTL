# Bayesian Uncertainty for Gradient Aggregation in Multi-Task Learning [ICML 2024]
As machine learning becomes more prominent there is a growing demand to perform several inference tasks in parallel. Running a dedicated model for each task is computationally expensive and therefore there is a great interest in multi-task learning (MTL). MTL aims at learning a single model that solves several tasks efficiently. Optimizing MTL models is often achieved by computing a single gradient per task and aggregating them for obtaining a combined update direction. However, these approaches do not consider an important aspect, the sensitivity in the gradient dimensions. Here, we introduce a novel gradient aggregation approach using Bayesian inference. We place a probability distribution over the task-specific parameters, which in turn induce a distribution over the gradients of the tasks. This additional valuable information allows us to quantify the uncertainty in each of the gradients dimensions, which can then be factored in when aggregating them. We empirically demonstrate the benefits of our approach in a variety of datasets, achieving state-of-the-art performance.

[[Paper]](https://arxiv.org/abs/2402.04005)

### Installation Instructions
1. Install repo:
```bash
conda create -n "BayesAgg-MTL" python=3.9
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
```
2. Download the UTKFace dataset from the offical repository [[Link]](https://susanqq.github.io/UTKFace/) or kaggle [[Link]](https://www.kaggle.com/datasets/jangedoo/utkface-new) and place it under experiments/utkface/dataset
3. Download the ChestX-Ray14 dataset from kaggle [[Link]](https://www.kaggle.com/datasets/nih-chest-xrays/data) and place it under experiments/ChestX_ray14/dataset

### Running the code
```bash
cd experiments/xxx
python trainer.py
```
Where xxx is {utkface, ChestX_ray14, CIFAR_MTL}

### Citation
Please cite this paper if you want to use it in your work,
```
@inproceedings{achituve2024bayesian,
  title={Bayesian uncertainty for gradient aggregation in multi-task learning},
  author={Achituve, Idan and Diamant, Idit and Netzer, Arnon and Chechik, Gal and Fetaya, Ethan},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={117--134},
  year={2024}
}
```
