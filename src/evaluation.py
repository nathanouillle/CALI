import os
from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from .constants import ARCHI_LAYERS, DEVICE, MODELS_DIR, RESULTS_DIR
from .model import create_model
from .extractor import run_full_extraction_pipeline
from .trustscore_baseline import TrustScore
from .utils import (
    compute_aurc_eaurc,
    compute_aupr_score,
    compute_fpr_at_95_tpr,
    compute_wasserstein_score,
    get_loader_cifar10,
    normalise,
)


def to_numpy(x: Union[torch.Tensor, np.ndarray, ArrayLike]) -> np.ndarray:
    """Convert tensor-like input to a flattened NumPy array of class indices."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x_arr = np.asarray(x)
    if x_arr.ndim > 1:
        return x_arr.argmax(axis=1) if x_arr.shape[1] > 1 else x_arr.flatten()
    return x_arr.flatten()


def load_cali_scores(seed: int) -> Optional[np.ndarray]:
    """Load WAE-KDE scores from the penultimate layer."""
    arch_config = ARCHI_LAYERS.get("resnet")
    layer = arch_config[-2] if arch_config else None
    if layer is None:
        return None

    path = f"{RESULTS_DIR}/cwae_cifar10_resnet/seed{seed}/{layer}/eval_scores.npz"
    if not os.path.exists(path):
        return None

    data = np.load(path)
    return np.asarray(data["scores"])


def get_baseline_scores(
    eval_features: dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MSP softmax confidence and normalized energy-based confidence."""
    logits = eval_features["logits"].cpu()
    probs = F.softmax(logits, dim=1)
    softmax_scores = torch.max(probs, dim=1).values.numpy()

    energy_scores = -np.asarray(logsumexp(logits.numpy(), axis=1), dtype=np.float64)
    return normalise(softmax_scores), normalise(-energy_scores)


def compute_metrics(
    scores: ArrayLike, labels: ArrayLike
) -> Tuple[float, float, float, float, float, float, float]:
    """Compute confidence-quality metrics from scores and binary correctness labels."""
    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    auroc = float(roc_auc_score(labels_arr, scores_arr))
    fpr95 = float(compute_fpr_at_95_tpr(scores_arr, labels_arr))
    aupr = float(compute_aupr_score(scores_arr, labels_arr))
    aupr_error = float(compute_aupr_score(-scores_arr, 1 - labels_arr))
    wasserstein = float(compute_wasserstein_score(scores_arr, labels_arr))
    aurc, eaurc = compute_aurc_eaurc(scores_arr, labels_arr)

    return auroc, fpr95, aupr, aupr_error, wasserstein, float(aurc), float(eaurc)


def enable_dropout(model: torch.nn.Module) -> None:
    """Set dropout modules to train mode while keeping the model in eval mode."""
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def mc_dropout_inference(
    model: torch.nn.Module,
    test_loader,
    T: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run MC Dropout inference and return confidence and correctness arrays."""
    model.eval()
    enable_dropout(model)

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="MC Dropout"):
            inputs = inputs.to(DEVICE)
            batch_probs: list[torch.Tensor] = []

            for _ in range(T):
                try:
                    outputs = model(inputs, mc_dropout=True)
                except TypeError:
                    outputs = model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                batch_probs.append(F.softmax(logits, dim=1))

            all_probs.append(torch.stack(batch_probs).mean(dim=0).cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0).flatten()

    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    confidence = -entropy

    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == labels).astype(int)

    return confidence, correctness


def get_mc_dropout(seed: int, nb_inference: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Get MC Dropout confidence from cache or by computing fresh predictions."""
    cache_dir = os.path.join(RESULTS_DIR, "mc_dropout_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"mc_cifar10_resnet_seed{seed}_n{nb_inference}.npz"
    )

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return np.asarray(data["confidence"]), np.asarray(data["correctness"])

    print(f"Computing MC Dropout (N={nb_inference}) for cifar10...")

    _, eval_loader, num_classes = get_loader_cifar10()
    model = create_model(num_classes=num_classes)

    ckpt_path = os.path.join(MODELS_DIR, f"cifar10_resnet_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)

    confidence, correctness = mc_dropout_inference(model, eval_loader, nb_inference)
    confidence = normalise(confidence)

    np.savez_compressed(cache_path, confidence=confidence, correctness=correctness)

    return confidence, correctness


def compute_deep_ensemble(
    ensemble_seeds: list[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute deep-ensemble entropy confidence and correctness.

    The function loads eval logits for multiple seeds, averages class
    probabilities, and returns entropy-based confidence (`-H`) along with
    correctness labels of the ensemble predictions.
    """
    if len(ensemble_seeds) < 2:
        raise ValueError("At least 2 seeds are required for deep ensemble computation.")

    all_probs = []
    labels_tensor = None

    for seed in ensemble_seeds:
        path = f"{RESULTS_DIR}/features_cifar10_resnet/seed{seed}/eval_features.pt"
        if not os.path.exists(path):
            run_full_extraction_pipeline(seed=seed)

        data = torch.load(path)
        logits_tensor = data["logits"].float().cpu()
        probs_tensor = F.softmax(logits_tensor, dim=1)
        all_probs.append(probs_tensor)

        if labels_tensor is None:
            labels_tensor = data["labels"].cpu()
            if labels_tensor.dim() > 1:
                if labels_tensor.shape[-1] == 1:
                    labels_tensor = labels_tensor.squeeze(-1)
                else:
                    print(
                        "Warning: labels have unexpected shape "
                        f"{tuple(labels_tensor.shape)}. Using first column."
                    )
                    labels_tensor = labels_tensor[:, 0]

    if not all_probs:
        raise ValueError("No model outputs were loaded for deep ensemble.")
    if labels_tensor is None:
        raise ValueError("Unable to load labels for deep ensemble.")

    stacked_probs = torch.stack(all_probs, dim=0)
    mean_probs = torch.mean(stacked_probs, dim=0)

    ensemble_preds = torch.argmax(mean_probs, dim=1).cpu().numpy()
    labels = labels_tensor.cpu().numpy()
    ensemble_correctness = (ensemble_preds == labels).astype(int)

    epsilon = 1e-12
    ensemble_entropy = (
        -torch.sum(
            mean_probs * torch.log(mean_probs + epsilon),
            dim=1,
        )
        .cpu()
        .numpy()
    )
    deep_ensemble_scores = normalise(-ensemble_entropy)

    return deep_ensemble_scores, ensemble_correctness


def evaluate_all_methods(
    seed: int = 0, deep_ensemble_seeds: list[int] = [0, 1, 2, 3, 42]
) -> pd.DataFrame:
    """Evaluate all confidence methods and return a unified results DataFrame."""

    feat_path = f"{RESULTS_DIR}/features_cifar10_resnet/seed{seed}"
    train_path = f"{feat_path}/train_features.pt"
    eval_path = f"{feat_path}/eval_features.pt"
    confidnet_path = f"{RESULTS_DIR}/confidnet_cifar10_resnet_seed{seed}.npz"
    if not (
        os.path.exists(eval_path)
        or os.path.exists(train_path)
        or os.path.exists(confidnet_path)
    ):
        raise FileNotFoundError(
            f"Missing required files for seed {seed}. Ensure features and ConfidNet scores are computed."
        )

    tuning_features = torch.load(train_path)
    eval_features = torch.load(eval_path)

    labels_tuning_1d = to_numpy(tuning_features["labels"])
    labels_eval_1d = to_numpy(eval_features["labels"])
    preds_eval_1d = to_numpy(eval_features["predictions"])

    cali_scores = load_cali_scores(seed)
    softmax_scores, energy_scores = get_baseline_scores(eval_features)
    confidnet_scores = normalise(np.load(confidnet_path)["confidences"])
    mc_dropout_scores, correctness_mc_dropout = get_mc_dropout(
        seed=seed, nb_inference=50
    )
    deep_ensemble_scores, deep_correctness = compute_deep_ensemble(
        ensemble_seeds=deep_ensemble_seeds
    )

    layers = ARCHI_LAYERS.get("resnet")
    if layers is None or len(layers) < 2:
        raise ValueError(f"Architecture 'resnet' is not valid in ARCHI_LAYERS.")

    # Trust Score scoring from the penultimate layer and hyperparameters used in the original paper: https://github.com/google/TrustScore/tree/master
    trust_layer = layers[-2]
    X_tuning = tuning_features[trust_layer].numpy()
    X_eval = eval_features[trust_layer].numpy()

    pca = PCA(n_components=20)
    X_tuning = pca.fit_transform(X_tuning)
    X_eval = pca.transform(X_eval)

    trust_scorer = TrustScore(k=10, alpha=0)
    trust_scorer.fit(X_tuning, labels_tuning_1d)
    trust_scores = normalise(trust_scorer.get_score(X_eval, preds_eval_1d))

    data = {
        "gold_label": labels_eval_1d,
        "predicted_label": preds_eval_1d,
        "score_cali": cali_scores,
        "score_confidnet": confidnet_scores,
        "score_softmax": softmax_scores,
        "score_energy": energy_scores,
        "score_mcdropout": mc_dropout_scores,
        "mc_dropout_correctness": correctness_mc_dropout,
        "deep_ensemble_scores": deep_ensemble_scores,
        "deep_correctness": deep_correctness,
        "score_trust": trust_scores,
    }

    return pd.DataFrame(
        {
            key: (value.ravel() if hasattr(value, "ravel") else value)
            for key, value in data.items()
            if value is not None
        }
    )


def compute_all_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Compute the metrics table for all evaluated confidence methods."""
    correctness = (results["gold_label"] == results["predicted_label"]).astype(int)

    metrics_cali = compute_metrics(scores=results["score_cali"], labels=correctness)
    metrics_softmax = compute_metrics(
        scores=results["score_softmax"], labels=correctness
    )
    metrics_trust = compute_metrics(scores=results["score_trust"], labels=correctness)
    metrics_energy = compute_metrics(scores=results["score_energy"], labels=correctness)
    metrics_confidnet = compute_metrics(
        scores=results["score_confidnet"], labels=correctness
    )
    metrics_mc_dropout = compute_metrics(
        scores=results["score_mcdropout"],
        labels=results["mc_dropout_correctness"],
    )
    metrics_deep_ensemble = compute_metrics(
        scores=results["deep_ensemble_scores"],
        labels=results["deep_correctness"],
    )

    row_titles = [
        "AUROC",
        "FPR@95TPR (lower)",
        "AUPR",
        "AUPR-error",
        "Wasserstein",
        "AURC (lower)",
        "e-AURC (lower)",
    ]

    return pd.DataFrame(
        {
            "WAE_Cali": metrics_cali,
            "Softmax": metrics_softmax,
            "TrustScore": metrics_trust,
            "EnergyScore": metrics_energy,
            "MC_Dropout": metrics_mc_dropout,
            "Deep_Ensemble": metrics_deep_ensemble,
            "ConfidNet": metrics_confidnet,
        },
        index=row_titles,
    )


def compare_methods(
    seed: int = 0, deep_ensemble_seeds: list[int] = [0, 1, 2, 3, 42]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full evaluation pipeline and print the metrics comparison."""
    if len(deep_ensemble_seeds) < 2:
        raise ValueError("At least 2 seeds are required for deep ensemble evaluation.")
    results = evaluate_all_methods(seed=seed, deep_ensemble_seeds=deep_ensemble_seeds)
    metrics_table = compute_all_metrics(results)

    print("\n--- Confidence Metrics Comparison ---")
    print(metrics_table.round(4))
    return metrics_table, results


def analyze_failure_threshold(
    df: pd.DataFrame,
    score_col: str,
    n_points: int = 50,
    delta_ratio: float = 0.05,
) -> dict[str, Any]:
    """Analyze local robustness of a score threshold around its ROC-optimal value.

    The function computes an optimal acceptance threshold (`tau_star`) from the
    ROC curve (Youden criterion), then evaluates how the accepted-sample error
    risk changes in a local neighborhood around `tau_star`.

    Args:
        df: Evaluation DataFrame containing labels, predictions and score columns.
        score_col: Name of the confidence-score column to analyze.
        n_points: Number of thresholds sampled in the local neighborhood.
        delta_ratio: Neighborhood half-width as a ratio of score range.

    Returns:
        Dictionary with:
            - `tau_star`: best threshold maximizing TPR - FPR.
            - `risk_star`: empirical risk at `tau_star`.
            - `test_thresholds`: sampled thresholds around `tau_star`.
            - `risk_increase`: relative degradation of risk vs `risk_star`.
            - `ari`: average risk increase over the local neighborhood.

    Raises:
        KeyError: If `score_col` does not exist in `df`.
        ValueError: If `n_points < 2` or `delta_ratio < 0`.
    """
    if score_col not in df.columns:
        raise KeyError(f"Column '{score_col}' not found in DataFrame.")
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")
    if delta_ratio < 0:
        raise ValueError("delta_ratio must be >= 0.")

    if score_col == "deep_ensemble_scores" and "deep_correctness" in df.columns:
        y_error = 1 - df["deep_correctness"].to_numpy()
    elif score_col == "score_mcdropout" and "mc_dropout_correctness" in df.columns:
        y_error = 1 - df["mc_dropout_correctness"].to_numpy()
    else:
        y_error = (df["gold_label"] != df["predicted_label"]).astype(int).to_numpy()

    scores = df[score_col].to_numpy()

    fpr, tpr, thresholds = roc_curve(y_error, -scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    tau_star = float(-thresholds[best_idx])

    def compute_risk(threshold: float) -> float:
        accepted = scores >= threshold
        if accepted.sum() == 0:
            return 0.0
        return float(np.mean(y_error[accepted]))

    risk_star = compute_risk(tau_star)

    score_range = float(scores.max() - scores.min())
    delta = float(delta_ratio * score_range)
    test_thresholds = np.linspace(tau_star - delta, tau_star + delta, n_points)

    risks = np.array([compute_risk(float(threshold)) for threshold in test_thresholds])

    epsilon = 1e-6
    risk_increase = np.abs(risks - risk_star) / (risk_star + epsilon)
    ari = float(np.trapezoid(risk_increase, test_thresholds))

    width = float(test_thresholds[-1] - test_thresholds[0])
    if width > 0:
        ari /= width

    return {
        "tau_star": tau_star,
        "risk_star": risk_star,
        "test_thresholds": test_thresholds,
        "risk_increase": risk_increase,
        "ari": ari,
    }


def get_failure_threshold(
    seed: int,
    deep_ensemble_seeds: list[int],
    delta: float,
) -> pd.DataFrame:
    """Compute and save local threshold-robustness metrics for one seed.

    The function runs a single evaluation pass, computes failure-threshold
    robustness metrics (including ARI) for every detected score column and every
    delta value, then saves the consolidated table as CSV.

    Args:
        seed: Random seed used to load model/features and run evaluation.
        deep_ensemble_seeds: List of seeds for deep ensemble methods.
        delta: Neighborhood ratio passed to `analyze_failure_threshold`.

    Returns:
        The threshold analysis DataFrame for the given seed and delta.
    """
    df = evaluate_all_methods(seed=seed, deep_ensemble_seeds=deep_ensemble_seeds)
    score_cols = [col for col in df.columns if "score_" in col or "_scores" in col]

    seed_results: list[dict[str, Any]] = []
    for score_col in score_cols:
        analysis = analyze_failure_threshold(
            df=df,
            score_col=score_col,
            delta_ratio=float(delta),
            n_points=50,
        )

        seed_results.append(
            {
                "Seed": seed,
                "Dataset": "cifar10",
                "Architecture": "resnet18",
                "Method": score_col,
                "Delta": float(delta),
                "ARI": analysis["ari"],
                "Tau Star": analysis["tau_star"],
                "Risk Star": analysis["risk_star"],
                "Risk Increase": analysis["risk_increase"],
                "Test Thresholds": analysis["test_thresholds"],
            }
        )
    return pd.DataFrame(seed_results)
