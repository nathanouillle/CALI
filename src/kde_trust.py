from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.special import expit as sigmoid
import pandas as pd
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance
from typing import Union, List, Dict, Any
import torch


class KDETrust:
    def __init__(
        self,
        kernel: str = "exponential",
        bandwidth: Union[str, List[float]] = "silverman",
        distance_metric: str = "euclidean",
    ):
        self.bandwidth_grid = bandwidth
        self.kernel = kernel
        self.distance_metric: str = distance_metric
        self.kdes_good: Dict[int, KernelDensity] = {}
        self.kdes_bad: Dict[int, KernelDensity] = {}
        self.class_labels: List[int] = []
        self.bandwidths_good: Dict[int, float] = {}
        self.bandwidths_bad: Dict[int, float] = {}
        self.priors_good: Dict[int, float] = {}
        self.priors_bad: Dict[int, float] = {}
        self.results: pd.DataFrame = pd.DataFrame()

    def _find_best_bandwidth(
        self, X: np.ndarray, bandwidths: Union[List[float], str]
    ) -> tuple:
        """
        Perform grid search on KDE bandwidth
        """
        kde_params = {"kernel": self.kernel, "metric": self.distance_metric}

        if self.distance_metric == "mahalanobis":
            metric_params = self._compute_metric_params(X)
            kde_params["metric_params"] = metric_params

        kde = KernelDensity(**kde_params)

        grid = GridSearchCV(kde, {"bandwidth": bandwidths}, cv=2)
        grid.fit(X)
        return grid.best_estimator_, grid.best_params_["bandwidth"]

    def _compute_metric_params(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Compute the metric parameters (VI_good or VI_bad) for Mahalanobis distance.
        """
        if self.distance_metric != "mahalanobis":
            raise ValueError(
                "metric_params should only be computed for Mahalanobis distance."
            )

        ec = EmpiricalCovariance().fit(X)
        precision = ec.precision_
        return {"VI": precision}

    def _kde_def(self, X: np.ndarray) -> KernelDensity:
        """Create a KernelDensity object with the specified parameters."""
        kde_params = {
            "kernel": self.kernel,
            "metric": self.distance_metric,
            "bandwidth": self.bandwidth_grid,
        }

        if self.distance_metric == "mahalanobis":
            metric_params = self._compute_metric_params(X)
            kde_params["metric_params"] = metric_params

        return KernelDensity(**kde_params)

    def _fit_kdes(self, X: np.ndarray, cls: int, category: str) -> None:
        """
        Fit KDEs for a specific class and category ("GOOD" or "BAD").
        """
        kdes_dict = self.kdes_good if category == "GOOD" else self.kdes_bad
        bandwidths_dict = (
            self.bandwidths_good if category == "GOOD" else self.bandwidths_bad
        )

        if isinstance(self.bandwidth_grid, (list, np.ndarray)):
            kde, best_bw = self._find_best_bandwidth(X, self.bandwidth_grid)
        else:
            kde = self._kde_def(X).fit(X)
            best_bw = kde.bandwidth_

        kdes_dict[cls] = kde
        bandwidths_dict[cls] = best_bw

    def fit(self, X: np.ndarray, y: np.ndarray, correctness: np.ndarray) -> None:
        """
        Fit the KDEs based on class labels and correctness.

        Parameters:
        - X: (N, D) training embeddings or logits
        - y: (N,) predicted class labels
        - correctness: (N,) binary (1 = correct, 0 = incorrect)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(correctness, torch.Tensor):
            correctness = correctness.detach().cpu().numpy()
        self.class_labels = np.unique(y).tolist()
        for cls in tqdm(self.class_labels, desc="Fitting KDEs by correctness"):
            X_cls = X[y == cls]
            c_cls = correctness[y == cls]
            X_good = X_cls[c_cls == 1]
            X_bad = X_cls[c_cls == 0]

            if len(X_good) == 0 or len(X_bad) == 0:
                raise ValueError(
                    f"Cannot fit KDEs with empty data.\nThere are {len(X_good)} correct classification samples and {len(X_bad)} incorrect classification samples for class {cls}."
                )

            # Bayesian Priors
            N_good = len(X_good)
            N_cls = len(X_cls)
            self.priors_good[cls] = float(N_good) / (N_cls)
            self.priors_bad[cls] = 1 - self.priors_good[cls]

            # Fit KDEs for correct and incorrect instances
            self._fit_kdes(X_good, cls, "GOOD")
            self._fit_kdes(X_bad, cls, "BAD")

    def score(self, x: np.ndarray, y: int) -> Dict[str, Union[np.ndarray, float]]:
        """
        Score a single instance.

        Parameters:
        - x: (D,) test embedding or logit
        - y: predicted class label

        Returns:
        - confidence score in [0, 1]
        """
        if isinstance(y, torch.Tensor):
            y = y.item()
        y = int(y)
        if y not in self.kdes_good or y not in self.kdes_bad:
            raise ValueError(f"Predicted class {y} is not fitted.")

        log_p_good = self.kdes_good[y].score_samples(x.reshape(1, -1))[0]
        log_p_bad = self.kdes_bad[y].score_samples(x.reshape(1, -1))[0]
        diff = log_p_good - log_p_bad
        confidence = sigmoid(diff)

        # Bayesian prediction
        log_p_good_bay = log_p_good + np.log(self.priors_good[y])
        log_p_bad_bay = log_p_bad + np.log(self.priors_bad[y])
        confidence_bay = sigmoid((log_p_good_bay - log_p_bad_bay))

        return {
            "input": x.copy(),
            "confidence": confidence,
            "confidence_bay": confidence_bay,
        }

    def score_batch(self, X, y) -> pd.DataFrame:
        """
        Score a batch of instances.

        Parameters:
        - X: (N, D) test embeddings or logits
        - y: (N,) array of predicted class labels

        Returns:
        - DataFrame with confidence scores
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if len(X) != len(y):
            raise ValueError("Length of X and y must be the same.")
        scores = []
        self.results = pd.DataFrame()
        for x, cls in tqdm(zip(X, y), total=len(X), desc="KDE-Trust scoring"):
            score = self.score(x, cls)
            scores.append(score)
        self.results = pd.DataFrame(scores)
        return self.results
