import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from typing import Dict, List, Tuple

from .constants import RESULTS_DIR, MODELS_DIR, DEVICE
from .model import create_model
from .utils import get_loader_cifar10

TensorDict = Dict[str, torch.Tensor]

class FeatureExtractor:
    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        """Initialize a feature extractor with forward hooks on backbone layers."""
        self.model = model
        self.device = device
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[RemovableHandle] = []
        
        # Define layers to monitor.
        self.layer_mapping = self._get_layer_mapping()

    def _get_layer_mapping(self) -> Dict[str, nn.Module]:
        """Return the feature-name to module mapping for the current backbone. Only ResNet is implemented in this demo."""
        return {
            # For a lighter extraction, we only hook the final block.
            #'conv1': self.model.backbone.conv1,
            #'layer1': self.model.backbone.layer1,
            #'layer2': self.model.backbone.layer2,
            #'layer3': self.model.backbone.layer3,
            'layer4': self.model.backbone.layer4,
        }
        

    def _get_hook(self, name: str):
        """
        Build a forward hook that stores compact activations.

        - 4D tensors (N, C, H, W) are pooled to (N, C) via GAP.
        - Higher-rank tensors are flattened from dim 1.
        """
        def hook(_module: nn.Module, _input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            data = output
            
            # Use GAP for convolutional maps.
            if data.dim() == 4:
                data = F.adaptive_avg_pool2d(data, (1, 1))
                data = data.flatten(1)
            
            # Flatten non-2D tensors.
            elif data.dim() > 2:
                data = data.flatten(1)

            # Store detached CPU activations.
            self.activations[name] = data.detach().cpu()
        return hook

    def _register_hooks(self) -> None:
        """Attach forward hooks to all mapped layers."""
        self.hooks = []
        for name, layer in self.layer_mapping.items():
            self.hooks.append(layer.register_forward_hook(self._get_hook(name)))

    def _remove_hooks(self) -> None:
        """Remove all previously attached hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def extract(self, loader: DataLoader) -> TensorDict:
        """Run feature extraction over an entire dataloader."""
        results: Dict[str, List[torch.Tensor]] = {name: [] for name in self.layer_mapping.keys()}
        results.update({'logits': [], 'labels': [], 'predictions': []})

        self.model.eval()
        self._register_hooks()
        
        print("Extracting features for architecture: ResNet")
        
        try:
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)

                    # Keep logits if the model returns a tuple.
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    
                    for name in self.layer_mapping.keys():
                        if name not in self.activations:
                            raise RuntimeError(f"Missing activation for layer '{name}'.")
                        results[name].append(self.activations[name])
                    
                    results['logits'].append(logits.detach().cpu())
                    results['labels'].append(labels.detach().cpu())
                    results['predictions'].append(torch.argmax(logits, dim=1).detach().cpu())
        finally:
            self._remove_hooks()

        return {k: torch.cat(v) for k, v in results.items()}

def compute_accuracy(data: TensorDict) -> float:
    """Compute accuracy from extracted prediction tensors."""
    correct = (data['predictions'] == data['labels']).float()
    return correct.mean().item() * 100

def run_full_extraction_pipeline(seed: int) -> None:
    """Run end-to-end extraction for CIFAR-10 ResNet and save feature files."""
    save_path = os.path.join(RESULTS_DIR, f"features_cifar10_resnet", f"seed{seed}")
    train_features_path = os.path.join(save_path, "train_features.pt")
    eval_features_path = os.path.join(save_path, "eval_features.pt")
    if os.path.exists(train_features_path) and os.path.exists(eval_features_path):
        print(f"[SKIP EXTRACTION] cifar10 | seed {seed} - feature files already exist.")
        return
    
    # 1. Data loading.
    train_loader, eval_loader, num_classes = get_loader_cifar10()
    
    # 2. Model creation.
    model = create_model(num_classes=num_classes).to(DEVICE)

    ckpt_path = os.path.join(MODELS_DIR, f"cifar10_resnet_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"No trained model found at: {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    
    # 3. Extractor initialization.
    extractor = FeatureExtractor(model, DEVICE)
    
    # 4. Extraction.
    train_data = extractor.extract(train_loader)
    eval_data = extractor.extract(eval_loader)

    train_acc = compute_accuracy(train_data)
    eval_acc = compute_accuracy(eval_data)
    print("-" * 30)
    print("ResNet feature extraction results:")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Eval Accuracy:     {eval_acc:.2f}%")
    print("-" * 30)
    
    # 5. Save feature tensors.
    os.makedirs(save_path, exist_ok=True)
    torch.save(train_data, train_features_path)
    torch.save(eval_data, eval_features_path)
    
    print(f"Features saved to: {save_path}")