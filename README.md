# CALI

Anonymous demo repository for TML4DM-ICDM 2026.

CALI is a failure-prediction method for DNN classifiers. The pipeline combines:
- A tailored Cramer-Wold Auto-Encoder (CWAE) with the introduced push loss for latent space structuring,
- Confidence/non-confidence density estimation with KDE,
- Benchmark evaluation against standard confidence baselines.

## What is important in `src/`

- `src/cwae.py`: Tailored CWAE implementation (including the custom push loss).
- `src/kde_trust.py`: KDE-based density estimation for confidence/non-confidence zones.
- `src/run_cali.py`: End-to-end CALI pipeline (`CWAE -> KDE -> scoring`).

## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook CALI.ipynb
```

## Demo workflow (all in the notebook)

### 1. Prepare model checkpoints

Do this first. The notebook is intentionally lightweight and expects checkpoints to already exist.

You have two options:

- **Option A (recommended for quick demo):** download **all** classifier checkpoints from **[[HERE](https://osf.io/k2tgw/overview?view_only=b53979387b844dd195832d3cb9a9bbfa)]** and unzip them the root of this repository. The expected files are:
	- `models/cifar10_resnet_seed<seed>.pth`
	- `results/confidnet_cifar10_resnet_seed<seed>.npz`

It will allow you to run the full notebook pipeline without waiting for training the models and other baselines.
- **Option B (train locally):** follow these training steps before running the full notebook pipeline:
	- Train classifiers for the seeds you use in the Deep Ensemble baseline (for example `[0, 1, 2, 3, 42]`) with `train_model(seed=seed)`.
	- Train ConfidNet with `train_confidnet(seed=seed)`.
	- Evaluate ConfidNet with `evaluate_confidnet(seed=seed)`.

### 2. Run the notebook pipeline

It will:
- Run the full CALI pipeline with `run_cali(seed=seed)`.
- Evaluate and compare all methods with `compare_methods(seed=seed, deep_ensemble_seeds=deep_ensemble_seeds)`.
- Plot the separation and operational stability failure curves.


## Notes

- Dataset used in this demo: CIFAR-10.
- Main usage is notebook-first (`CALI.ipynb`).
- Code for Trust Score borrowed from : [HERE](https://github.com/google/TrustScore) ([LICENSE](https://github.com/google/TrustScore/blob/master/LICENSE))
- Code for ConfidNet borrowed from : [HERE](https://github.com/valeoai/ConfidNet) ([LICENSE](https://github.com/valeoai/ConfidNet/tree/master/LICENSE))

## Detailed Results on Datasets

The following tables present detailed performance metrics for several architecture-dataset pairs, reporting mean and standard deviation across 5 independent runs with different seeds.

For the Deep Ensemble method, standard deviation is not applicable as 5 runs are combined into one result.

**Bold values** indicate the best-performing method per metric (within a tolerance of $\pm 0.001$ to account for numerical precision). 
<img width="1313" height="295" alt="image" src="figures/CIFAR10 -- ResNet-18.png" />
<img width="1307" height="293" alt="image" src="figures/CIFAR10 -- VGG-16.png" />
<img width="1311" height="295" alt="image" src="figures/CIFAR100 -- ResNet-50.png" />
<img width="1310" height="295" alt="image" src="figures/FashionMNIST -- CNN.png" />
<img width="1316" height="285" alt="image" src="figures/MNIST -- CNN.png" />
<img width="1308" height="279" alt="image" src="figures/PneumoniaMNIST -- ResNet-18.png" />
<img width="1309" height="294" alt="image" src="figures/SVHN -- DenseNet-121.png" />
<img width="1307" height="287" alt="image" src="figures/SVHN -- WideResNet-50.png" />


### Datasets Detailed Statistics
Details the per-class sample distribution for each dataset. It reports the average and minimum number of correct and incorrect predictions available to fit the failure prediction methods, conditioned on the classifier's performance.

<img width="1334" height="372" alt="image" src="figures/Dataset Statistics.png" />

### Time Complexity Example
Performed on a NVIDIA  Tesla  V100  GPU

<img width="708" height="349" alt="image" src="figures/Time Complexity.png" />

## Push Loss for Failure Prediction

In this work, we introduce a novel dimensionality reduction method, a CWAE regularized by a custom Push Loss, to exploit high-dimensional DNN embeddings for failure prediction. The following table demonstrates its impact:

<img width="1482" height="763" alt="image" src="figures/Dimensionality Reduction Ablation.png" />
