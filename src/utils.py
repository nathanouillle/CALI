import random
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
import pandas as pd
from typing import Tuple
from numpy.typing import ArrayLike, NDArray

from .constants import BATCH_SIZE, DATA_DIR


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_loader_cifar10() -> tuple[DataLoader, DataLoader, int]:
    """Load CIFAR-10 and return train/eval dataloaders with standard augmentation."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = datasets.CIFAR10(
        DATA_DIR, train=True, download=True, transform=train_transform
    )
    eval_set = datasets.CIFAR10(
        DATA_DIR, train=False, download=True, transform=eval_transform
    )
    num_classes = len(train_set.classes)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, eval_loader, num_classes


def compute_wasserstein_score(scores: ArrayLike, labels: ArrayLike) -> float:
    """Compute Wasserstein (Earth Mover's) distance between label-0 and label-1 scores."""
    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)
    s0 = scores_arr[labels_arr == 0]
    s1 = scores_arr[labels_arr == 1]
    return wasserstein_distance(s0, s1)


def compute_aurc_eaurc(scores: ArrayLike, labels: ArrayLike) -> Tuple[float, float]:
    """Compute AURC and E-AURC, scaled by 1000 for readability."""
    errors = 1 - np.asarray(labels)
    scores_arr = np.asarray(scores)

    # Sort by descending confidence.
    desc_score_indices = np.argsort(scores_arr)[::-1]
    sorted_errors = errors[desc_score_indices]

    # Cumulative risk at each coverage level.
    n = len(sorted_errors)
    cum_errors = np.cumsum(sorted_errors)
    risk_values = cum_errors / np.arange(1, n + 1)

    aurc = np.mean(risk_values)

    # Optimal AURC when all errors are ranked last.
    sorted_errors_opt = np.sort(errors)
    risk_values_opt = np.cumsum(sorted_errors_opt) / np.arange(1, n + 1)
    aurc_opt = np.mean(risk_values_opt)

    e_aurc = aurc - aurc_opt

    return aurc * 1000, e_aurc * 1000


def normalise(x: ArrayLike, eps: float = 1e-12) -> NDArray[np.float64]:
    """Normalize a vector to [0, 1] while handling NaN and infinity values."""
    # Convert safely to numeric values (non-numeric entries become NaN).
    x_numeric = np.asarray(
        pd.to_numeric(pd.Series(x), errors="coerce"), dtype=np.float64
    )

    # Identify finite values only.
    mask_finite = np.isfinite(x_numeric)

    if not np.any(mask_finite):
        return np.zeros_like(x_numeric)

    # Compute min/max on finite values only.
    finite_values = x_numeric[mask_finite]
    true_max = np.max(finite_values)
    true_min = np.min(finite_values)

    # Replace invalid values with finite bounds.
    x_clean = np.nan_to_num(x_numeric, nan=true_min, posinf=true_max, neginf=true_min)

    # Standard min-max normalization.
    min_val = x_clean.min()
    max_val = x_clean.max()
    denom = max_val - min_val

    if denom < eps:
        return np.zeros_like(x_clean)

    return (x_clean - min_val) / denom


def compute_fpr_at_95_tpr(scores: ArrayLike, labels: ArrayLike) -> float:
    """Compute FPR at the operating point where TPR is closest to 95%."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(fpr[np.argmin(np.abs(tpr - 0.95))])


def compute_aupr_score(scores: ArrayLike, labels: ArrayLike) -> float:
    """Compute area under the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))
