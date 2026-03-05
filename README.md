# CALI

Anonymous demo repository for ECML PKDD 2026.

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

Run the notebook cells in order.

### 1) Get a classifier checkpoint

You have two options:

- **Option A (train locally):** run `train_model(seed=seed)`.
- **Option B (recommended for quick demo):** download checkpoints from **[[HERE](https://osf.io/k2tgw/overview?view_only=b53979387b844dd195832d3cb9a9bbfa)]** and place them in `models/` with names like:
	- `cifar10_resnet_seed0.pt`
	- `cifar10_resnet_seed1.pt`
	- etc.

### 2) Extract features

Run:

```python
run_full_extraction_pipeline(seed=seed)
```

This extracts penultimate features + logits/predictions/labels and saves them in:
- `results/features_cifar10_resnet/seed{seed}/`

### 3) Run CALI (CWAE + KDE)

Run:

```python
run_cali(seed=seed)
```

This trains/infers CALI and writes scores/metrics under `results/`.

### 4) Compare with baselines

Run:

```python
metrics_table, detailed_results = compare_methods(seed=seed)
```

Compared methods include CALI, TrustScore, Softmax/MSP, Energy score, and MC Dropout.

### 5) Plot ARI (Threshold Robustness Metric)

Generate and visualise ARI (Average Risk Increase), introduced for threshold robustness:

```python
save_failure_threshold(seed=seed)
plot_operational_stability_failure(seed=seed, delta=0.05)
```

## Notes

- Dataset used in this demo: CIFAR-10.
- If checkpoints/features already exist, parts of the pipeline are skipped automatically.
- Main usage is notebook-first (`CALI.ipynb`).