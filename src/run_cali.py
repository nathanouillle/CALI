import torch
import os
import numpy as np
import time
import json
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass

from .evaluation import compute_metrics
from .cwae import MultiClassCWAEManager
from .kde_trust import KDETrust
from .extractor import run_full_extraction_pipeline
from .constants import CWAE_CONFIG, RESULTS_DIR, ARCHI_LAYERS

@dataclass
class PipelineResult:
    z_tuning: np.ndarray
    p_tuning: np.ndarray
    y_tuning: np.ndarray
    c_tuning: np.ndarray
    z_eval: np.ndarray
    p_eval: np.ndarray
    y_eval: np.ndarray
    c_eval: np.ndarray
    training_time: float
    z_tuning_recon_loss: float
    z_eval_recon_loss: float

def flatten_features(features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Flatten feature tensors from (N, C, H, W) to (N, D) when needed."""
    if len(features.shape) > 2:
        return features.reshape(features.shape[0], -1)
    return features

def to_numpy(tensor_or_array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert tensor-like input to a NumPy array."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().detach().numpy()
    return np.asarray(tensor_or_array)

def load_raw_data_dicts(seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load raw feature dictionaries from disk once.
    """
    folder = os.path.join(RESULTS_DIR, f"features_cifar10_resnet", f"seed{seed}")
    eval_path = os.path.join(folder, "eval_features.pt")
    train_path = os.path.join(folder, "train_features.pt")
    
    if not (os.path.exists(train_path) and os.path.exists(eval_path)):
        print("Extracting DNN features for the first time...")
        run_full_extraction_pipeline(seed=seed)


    eval_dict = torch.load(eval_path)
    train_dict = torch.load(train_path)

    return train_dict, eval_dict

def extract_layer_tensors(
    tuning_dict: Dict[str, Any],
    eval_dict: Dict[str, Any],
    layer_key: str,
) -> Tuple[
    Tuple[Union[np.ndarray, torch.Tensor], np.ndarray, np.ndarray, np.ndarray],
    Tuple[Union[np.ndarray, torch.Tensor], np.ndarray, np.ndarray, np.ndarray],
]:
    """Extract layer-specific features and compute correctness flags for two splits."""
    if layer_key not in tuning_dict:
        raise KeyError(
            f"Layer '{layer_key}' does not exist in extracted features. Available keys: {list(tuning_dict.keys())}"
        )

    # Internal helper to avoid duplicated split-processing logic.
    def process_split(data_dict: Dict[str, Any]) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
        feats = flatten_features(data_dict[layer_key])
        preds = to_numpy(data_dict['predictions'])
        labels = to_numpy(data_dict['labels']).astype(np.int64).flatten()
        if preds.ndim > 1:
            preds = np.argmax(preds, axis=1)
        else:
            preds = preds.astype(np.int64).flatten()
        # Correctness coding: 1=correct, 0=incorrect.
        correctness = (preds == labels).astype(np.float32)
        return feats, preds, labels, correctness

    # Extract train/tuning and eval tensors.
    X_train, p_train, y_train, c_train = process_split(tuning_dict)
    X_eval, p_eval, y_eval, c_eval = process_split(eval_dict)

    return (X_train, p_train, y_train, c_train), (X_eval, p_eval, y_eval, c_eval)

def run_pipeline_for_layer(
    seed: int,
    layer_key: str,
    raw_tuning: Dict[str, Any],
    raw_eval: Dict[str, Any],
    latent_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    lambda_cw: float,
    gamma_metric: float,
    margin_factor: float,
) -> PipelineResult:
    """
    Run the full CWAE pipeline for a single layer.
    """

    # 1. Extract split-specific tensors.
    (X_tuning, p_tuning, y_tuning, c_tuning), (X_eval, p_eval, y_eval, c_eval) = \
        extract_layer_tensors(raw_tuning, raw_eval, layer_key)
    
    
    input_dim = X_tuning.shape[1]
    print(f"Input Dim: {input_dim} | Train Samples: {len(X_tuning)}")

    # 2. Initialize manager.
    manager = MultiClassCWAEManager(
        input_dim=input_dim, 
        latent_dim=latent_dim, 
        seed=seed,
    )

    # 3. Training.
    print(f"Training CWAE ({CWAE_CONFIG['epochs']} epochs)...")
    start_time = time.time()
    manager.train_all_classes(
        feats=X_tuning,
        labels=y_tuning,
        predictions=p_tuning,
        correctness=c_tuning,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        lambda_cw=lambda_cw,
        gamma_metric=gamma_metric,
        visualise_latentspace=CWAE_CONFIG['visualise'],
        margin_factor=margin_factor,
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    # 4. Inference.
    z_tuning, z_tuning_recon_loss = manager.infer_latent_space(X_tuning, p_tuning, True)
    z_eval, z_eval_recon_loss = manager.infer_latent_space(X_eval, p_eval, True)

    return PipelineResult(
        z_tuning=z_tuning,
        p_tuning=p_tuning,
        y_tuning=y_tuning,
        c_tuning=c_tuning,
        z_eval=z_eval,
        p_eval=p_eval,
        y_eval=y_eval,
        c_eval=c_eval,
        training_time=training_time,
        z_tuning_recon_loss=z_tuning_recon_loss,
        z_eval_recon_loss=z_eval_recon_loss,
    )

def score_kde(
    pipeline: PipelineResult,
    seed: int,
    layer: str,
    save: bool,
    kernel: str,
    bandwidth: Union[str, List[float]],
    distance_metric: str,
) -> Tuple[Any, ...]:
    start_time = time.time()
    save_dir = os.path.join(RESULTS_DIR, f"cwae_cifar10_resnet", f"seed{seed}", layer)
    kde_scorer = KDETrust(
        kernel=kernel, bandwidth=bandwidth, distance_metric=distance_metric
    )

    kde_scorer.fit(
        X=pipeline.z_tuning,
        y=pipeline.p_tuning,
        correctness=pipeline.c_tuning,
    )

    eval_scores_dict = kde_scorer.score_batch(pipeline.z_eval, pipeline.p_eval)
    end_time = time.time()
    confidence_scores = eval_scores_dict['confidence']
    
    # Save scores and compute metrics.
    if save:
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(
            f"{save_dir}/eval_scores.npz",
            scores=confidence_scores,
            scores_bay=eval_scores_dict.get('confidence_bay', None),
            correctness=pipeline.c_eval,
            preds=pipeline.p_eval,
            labels=pipeline.y_eval,
            cwae_time=pipeline.training_time,
            kde_time=(end_time-start_time),
        )

    metrics = compute_metrics(scores=confidence_scores, labels=pipeline.c_eval)
    print(f"--- CALI metrics cifar10_resnet | {layer} ---")
    print(metrics)
    if save:
        results = {
            'auroc': metrics[0], 
            'fpr95': metrics[1], 
            'aupr': metrics[2], 
            'aupr_error': metrics[3], 
            'wasserstein': metrics[4], 
            'aurc': metrics[5], 
            'eaurc': metrics[6],
            'z_tuning_recon_loss': pipeline.z_tuning_recon_loss,
            'z_eval_recon_loss': pipeline.z_eval_recon_loss,
        }

        def _to_serializable(value: Any) -> Any:
            if isinstance(value, np.generic):
                return value.item()
            return value

        results_clean = {k: _to_serializable(v) for k, v in results.items()}
        with open(f"{save_dir}/metrics.json", 'w') as f:
            json.dump(results_clean, f, indent=4)
    return metrics

def run_cali(
    seed: int,
    save: bool = True,
    latent_dim: int = CWAE_CONFIG['latent_dim'],
    batch_size: int = CWAE_CONFIG['batch_size'],
    epochs: int = CWAE_CONFIG['epochs'],
    lr: float = CWAE_CONFIG['lr'],
    lambda_cw: float = CWAE_CONFIG['lambda_cw'],
    gamma_metric: float = CWAE_CONFIG['gamma_metric'],
    margin_factor: float = CWAE_CONFIG['margin_factor'],
    kernel: str = "exponential",
    bandwidth: Union[str, List[float]] = "silverman",
    distance_metric: str = "mahalanobis",
) -> None:
    # 1. Global loading (heavy I/O performed once).
    raw_tuning, raw_eval = load_raw_data_dicts(seed)
    
    # 2. Limited to penultimate block for this demo.
    layer = ARCHI_LAYERS.get('resnet', [])[-2]
    try:
        save_dir = os.path.join(RESULTS_DIR, f"cwae_cifar10_resnet", f"seed{seed}", layer)
        if not os.path.exists(f"{save_dir}/eval_scores.npz"):
            pipeline = run_pipeline_for_layer(
                seed=seed, layer_key=layer, raw_tuning=raw_tuning, raw_eval=raw_eval, 
            latent_dim=latent_dim,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            lambda_cw=lambda_cw,
            gamma_metric=gamma_metric,
            margin_factor=margin_factor,)
            score_kde(
                pipeline=pipeline,
                seed=seed,
                layer=layer,
                save=save,
                kernel=kernel,
                bandwidth=bandwidth,
                distance_metric=distance_metric,
            )
        else:
            data = np.load(f"{save_dir}/eval_scores.npz")
            metrics = compute_metrics(scores=data['scores'], labels=data['correctness'])
            print("--- CALI metrics cifar10_resnet ---")
            print(metrics)

    except Exception as e:
        print(f"Critical error on layer {layer}: {e}")
        import traceback
        traceback.print_exc()
        return
