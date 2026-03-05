import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from .constants import DEVICE
from .utils import set_seed

class CWAE(nn.Module):
    def __init__(self, input_dim: int = 512, latent_dim: int = 10, hidden_dim: int = 256) -> None:
        """
        Args:
            input_dim: Input feature dimension (e.g., 512 for ResNet18).
            latent_dim: Target latent dimension for KDE (e.g., 10 or 20).
            hidden_dim: Hidden layer width used by encoder/decoder.
        """
        super(CWAE, self).__init__()

        # Keep a minimum hidden width to avoid under-parameterized mappings.
        self.hidden_dim = max(hidden_dim, 64)
        
        if input_dim <= 4:
            latent_dim = 2

        self.latent_dim = latent_dim
        # --- ENCODER (Projection Head) ---
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim, affine=False),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # --- DECODER ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Utility function for inference-only encoding."""
        with torch.no_grad():
            return self.encoder(x)

class CWAELosses:
    """Container for static CWAE loss functions."""
    
    @staticmethod
    def cw_distance(Z: torch.Tensor) -> torch.Tensor:
        """
        Compute the Cramer-Wold distance term on latent samples.

        Args:
            Z: Latent tensor with shape [N, D].
        """
        N, D = Z.shape
        N = float(N)
        D = float(D)

        y = (4.0 / (3.0 * N)) ** 0.4
        K = 1.0 / (2.0 * D - 3.0)

        # ||Zi - Zj||^2
        Z_i = Z.unsqueeze(0)
        Z_j = Z.unsqueeze(1)
        A1 = torch.sum((Z_i - Z_j) ** 2, dim=2)

        A = (1.0 / (N ** 2)) * torch.sum(1.0 / torch.sqrt(y + K * A1))

        # ||Zi||^2
        B1 = torch.sum(Z ** 2, dim=1)
        B = (2.0 / N) * torch.sum(1.0 / torch.sqrt(y + 0.5 + K * B1))

        return (1.0 / torch.sqrt(torch.tensor(1.0 + y))) + A - B

    @staticmethod
    def push_loss(
        z: torch.Tensor,
        is_correct: torch.Tensor,
        latent_dim: int,
        margin_factor: float = 2.5,
    ) -> torch.Tensor:
        threshold_radius = math.sqrt(latent_dim) * margin_factor

        z_incorrect = z[is_correct == 0]
        if len(z_incorrect) == 0:
            return torch.tensor(0.0, device=z.device)
        z_norms = torch.norm(z_incorrect, p=2, dim=1)
        
        push_loss = torch.nn.functional.relu(threshold_radius - z_norms) ** 2
        return torch.mean(push_loss)

class MultiClassCWAEManager:
    def __init__(self, input_dim: int, latent_dim: int = 10, seed: int = 0) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = DEVICE
        self.models: Dict[int, CWAE] = {}
        self.histories: Dict[int, Dict[str, List[float]]] = {}

        set_seed(seed)

    def _get_optimizer(
        self, model: nn.Module, learning_rate: float, epochs: int
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler.CosineAnnealingLR]:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        return optimizer, scheduler

    def _visualize_latent_space(self, cwae_model: CWAE, features: torch.Tensor, flags: torch.Tensor) -> None:
        """
        Display correct-sample density and incorrect-sample positions in latent space.
        """
        cwae_model.eval()
        
        # 1. Extract latent vectors z.
        with torch.no_grad():
            device = next(cwae_model.parameters()).device
            features = features.to(device)
            _, z = cwae_model(features)
            z = z.cpu().numpy()
            flags = flags.cpu().numpy()

        # Check mean distance to origin.
        dist_correct = np.linalg.norm(z[flags == 1], axis=1).mean()
        dist_incorrect = np.linalg.norm(z[flags == 0], axis=1).mean()
        
        print("\n--- Results ---")
        print(f"Mean Distance (Correct)   : {dist_correct:.4f}")
        print(f"Mean Distance (Incorrect) : {dist_incorrect:.4f}")
        
        # 2. Dimensionality reduction to 2D via PCA.
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z)
        
        # Split by correctness.
        z_correct = z_2d[flags == 1]
        z_incorrect = z_2d[flags == 0]
        
        # 3. Plotting.
        plt.figure(figsize=(10, 8))
        
        # A. KDE plot for correct samples (confidence region).
        sns.kdeplot(
            x=z_correct[:, 0], 
            y=z_correct[:, 1], 
            fill=True, 
            cmap="Greens", 
            thresh=0.05, 
            levels=15,
            alpha=0.7,
            label='Correct Density (Confidence)'
        )
        
        # B. Scatter plot for incorrect samples.
        plt.scatter(
            z_incorrect[:, 0], 
            z_incorrect[:, 1], 
            color='red', 
            marker=cast(Any, 'x'), 
            s=30, 
            alpha=0.8, 
            label='Incorrect Samples'
        )

        # C. Light scatter for correct points.
        plt.scatter(
            z_correct[:, 0], 
            z_correct[:, 1], 
            color='green', 
            s=5, 
            alpha=0.1
        )
        
        plt.title("CWAE Latent Space (PCA 2D)\nGreen=Confidence, Red=Error")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def train_cwae(
        self,
        features: torch.Tensor,
        flags: torch.Tensor,
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        lambda_cw: float = 10,
        gamma_metric: float = 0.1,
        margin_factor: float = 2.5,
    ) -> Optional[Tuple[CWAE, Dict[str, List[float]]]]:
        """
        Train a CWAE with a cosine annealing learning-rate schedule.
        """
        
        dataset = TensorDataset(features, flags)

        n_samples = len(dataset)
        if n_samples == 0:
            print("Empty class subset, skipping.")
            return
    
        # BatchNorm handling.
        if n_samples < batch_size:
            if n_samples == 1:
                print("Warning: only one sample, BatchNorm cannot train. Skipping.")
                return    
            current_batch_size = n_samples
            loader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True, drop_last=False)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        
        # Model initialization.
        model = CWAE(input_dim=self.input_dim, latent_dim=self.latent_dim).to(self.device)
        self.latent_dim = model.latent_dim
        optimizer, scheduler = self._get_optimizer(model, learning_rate, epochs)
        mse_loss = nn.MSELoss()
        history = {'loss': [], 'std_z': [], 'learning_rate': []}
        
        warmup_epochs = 5  
        model.train()
        print(f"--- CWAE training start (Dim {self.input_dim} -> {self.latent_dim}) | LR schedule: Cosine ---")
        
        for epoch in range(epochs):
            total_loss = 0
            epoch_cw = 0
            epoch_push = 0
            epoch_recon = 0
            z_list = []

            beta = min(1.0, (epoch + 1) / warmup_epochs)
            
            current_lambda_cw = lambda_cw * beta
            current_gamma_metric = gamma_metric * beta
            
            for batch_x, batch_flags in loader:
                batch_x, batch_flags = batch_x.to(self.device), batch_flags.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass.
                recon_x, z = model(batch_x)
                
                # 1. Reconstruction loss.
                l_recon = mse_loss(recon_x, batch_x)
                
                # 2. CW distance on correct samples only.
                l_cw = torch.tensor(0.0, device=self.device)
                z_corrects = z[batch_flags == 1]
                z_list.append(z_corrects.detach().cpu())
                if z_corrects.size(0) > 1:
                    l_cw = CWAELosses.cw_distance(z_corrects)
                
                # 3. Push loss on incorrect samples only.
                l_push = CWAELosses.push_loss(z, batch_flags, self.latent_dim, margin_factor)
                                    
                # Total loss.
                loss = l_recon + (current_lambda_cw * l_cw) + (current_gamma_metric * l_push)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                epoch_recon += l_recon.item()
                epoch_cw += l_cw.item()
                epoch_push += l_push.item()
                
            current_learning_rate = scheduler.get_last_lr()[0]
            scheduler.step()

            # Epoch statistics.
            avg_loss = total_loss / len(loader)
            if len(z_list) > 0:
                all_z = torch.cat(z_list)
                std_z = torch.std(all_z, dim=0).mean().item()
            else:
                std_z = 0.0

            history['loss'].append(avg_loss)
            history['std_z'].append(std_z)
            history['learning_rate'].append(current_learning_rate)
            
            # Detailed logging.
            if (epoch + 1) % 5 == 0:
                print(f"Ep [{epoch+1}/{epochs}] | learning_rate: {current_learning_rate:.1e} | "
                    f"Loss: {avg_loss:.4f} | "
                    f'Rec Loss: {epoch_recon/len(loader):.4f} | '
                    f"CW Loss: {epoch_cw/len(loader):.4f} | "
                    f"Push Loss: {epoch_push/len(loader):.4f}")

            if history['std_z'][-1] < 0.1:
                print("Warning: collapse detected (very low latent STD).")

        return model, history
    
    def train_all_classes(
        self,
        feats: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        correctness: Union[np.ndarray, torch.Tensor],
        predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        visualise_latentspace: bool = False,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, List[float]]]:
        """
        Train one CWAE per class.
        
        Grouping logic:
        - Prefer grouping by prediction (model belief) for confidence modeling.
        - If `predictions` is None, fallback to ground-truth labels.
        """
        classes = len(np.unique(labels))
        print(f"=== Starting multi-class training ({classes} models) ===")
        
        # If explicit predictions are unavailable, route by labels.
        grouping = predictions if predictions is not None else labels
        
        # Convert once to tensors to avoid repeated conversion in the loop.
        X_tensor = torch.as_tensor(feats, dtype=torch.float32)
        flags_tensor = torch.as_tensor(correctness, dtype=torch.float32)
        group_tensor = torch.as_tensor(grouping)
        
        for class_id in range(classes):
            print(f"\nProcessing class {class_id}...")
            
            # 1. Filter data for the current class.
            mask = (group_tensor == class_id)
            
            if mask.sum() == 0:
                print(f"Warning: no data found for class {class_id}. Skipping.")
                continue

            # Extract class-specific subsets.
            X_tensor_cls = X_tensor[mask]
            flags_tensor_cls = flags_tensor[mask]
            cls_correctness = correctness[mask]

            print(f" Feature shape: {X_tensor_cls.shape}, Flags: {flags_tensor_cls.shape}")
            print(f"  -> Samples: {len(X_tensor_cls)} (Correct: {(cls_correctness==1).sum()}, Errors: {(cls_correctness==0).sum()})")
            
            # 2. Training.
            train_result = self.train_cwae(
                features=X_tensor_cls, 
                flags=flags_tensor_cls,
                **kwargs
            )
            if train_result is None:
                continue
            model, history = train_result
            
            # 3. Store model and history.
            self.models[class_id] = model
            self.histories[class_id] = history

            # 4. Visualization.
            if visualise_latentspace:
                print(f"  -> Generating latent-space plot for class {class_id}...")
                try:
                    self._visualize_latent_space(model, X_tensor, flags_tensor)
                    plt.suptitle(f"Latent Space - Class {class_id}", fontsize=16)
                    plt.show()
                except Exception as e:
                    print(f"Visualization error: {e}")
        return self.histories

    def infer_latent_space(
        self,
        test_feats: Union[np.ndarray, torch.Tensor],
        test_preds: Union[np.ndarray, torch.Tensor],
        return_recon_loss: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Infer latent codes for test features based on predicted class routing.

        Args:
            test_feats: Test features.
            test_preds: Predicted class IDs for routing to class-specific CWAEs.
            return_recon_loss: Whether to also return mean reconstruction loss.

        Returns:
            Latent codes as a NumPy array, and optionally the mean reconstruction loss.
        """
        self._check_models()
        num_samples = test_feats.shape[0]
        z_all = torch.zeros((num_samples, self.latent_dim), device=self.device)
        
        test_feats = torch.as_tensor(test_feats, dtype=torch.float32, device=self.device)
        test_preds = torch.as_tensor(test_preds, device=self.device)

        recon_losses: List[float] = []
        total_samples = 0
        if return_recon_loss:
            recon_losses = []
            total_samples = 0
        
        # Optimized class-wise inference.
        print("\n--- Inference on test set ---")
        for class_id, model in self.models.items():
            mask = (test_preds == class_id)
            if mask.any():
                model.eval()
                with torch.no_grad():
                    x_batch = test_feats[mask]
                    z_batch = model.encoder(x_batch)
                    z_all[mask] = z_batch
                    
                    # Optional reconstruction loss.
                    if return_recon_loss:
                        x_recon = model.decoder(z_batch)
                        batch_recon_loss = torch.nn.functional.mse_loss(
                            x_recon, x_batch, reduction='mean'
                        )
                        recon_losses.append(batch_recon_loss.item() * x_batch.shape[0])
                        total_samples += x_batch.shape[0]
        
        if return_recon_loss:
            mean_recon_loss = sum(recon_losses) / total_samples if total_samples > 0 else 0.0
            return z_all.cpu().numpy(), mean_recon_loss
        
        return z_all.cpu().numpy()

    def _check_models(self) -> None:
        if not self.models:
            raise ValueError("No model has been trained yet. Run train_all_classes first.")