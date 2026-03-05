import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader

from .utils import set_seed
from .constants import ARCHI_LAYERS, DEVICE, MODELS_DIR, RESULTS_DIR

# CODE BASED ON THE OFFICIAL CONFIDNET IMPLEMENTATION AVAILABLE AT: https://github.com/valeoai/ConfidNet/tree/master?tab=readme-ov-file
# Under the Apache License, Version 2.0, Copyright 2019 Valeo

def get_confid_net_model(input_dim: int) -> nn.Module:
    """
    Build the ConfidNet head used to predict confidence from penultimate features.
    """
    class ConfidNet(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 400)
            self.fc2 = nn.Linear(400, 400)
            self.fc3 = nn.Linear(400, 400)
            self.fc4 = nn.Linear(400, 400)
            self.fc5 = nn.Linear(400, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            return torch.sigmoid(self.fc5(x))

    model = ConfidNet(input_dim=input_dim)
    model.to(DEVICE)
    return model


def get_confidnet_loader(feature_path: str, layer_key: str, train: bool) -> Tuple[DataLoader, int]:
    if train:
        shuffle = True
        train_path = os.path.join(feature_path, "train_features.pt")
        data = torch.load(train_path)
    else:
        shuffle = False
        data = torch.load(feature_path)

    if layer_key not in data:
        raise KeyError(f"Layer '{layer_key}' not found in file. Available keys: {list(data.keys())}")

    features = data[layer_key]
    if features.dim() > 2:
        features = features.flatten(1)

    logits = data["logits"]
    labels = data["labels"]

    if labels.dim() > 1:
        labels = labels.squeeze()

    probs = F.softmax(logits, dim=1)
    tcp = probs[torch.arange(len(labels)), labels].unsqueeze(1)
    assert tcp.shape == (features.shape[0], 1), f"Invalid TCP shape: {tcp.shape}"

    dataset = TensorDataset(features, tcp)
    return DataLoader(dataset, batch_size=64, shuffle=shuffle), features.shape[1]


def train_confidnet(seed: int,epochs: int=170,lr: float=0.1,momentum: float=0.9,weight_decay: float=0.0001) -> float:
    """
    Train ConfidNet on extracted features using TCP as regression target.
    """
    save_path = f"{MODELS_DIR}/confidnet_cifar10_resnet_seed{seed}.pt"

    if os.path.exists(save_path):
        print(f"ConfidNet already trained (found at {save_path}).")

    set_seed(seed)

    layers = ARCHI_LAYERS.get('resnet')
    tuning_loader_confid, input_dim = get_confidnet_loader(feature_path=f"{RESULTS_DIR}/features_cifar10_resnet/seed{seed}", layer_key=layers[-2], train=True)

    confidnet = get_confid_net_model(input_dim)
    optimizer = torch.optim.SGD(
        confidnet.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        for features, tcp in tuning_loader_confid:
            features, tcp = features.to(DEVICE), tcp.to(DEVICE)
            conf_pred = confidnet(features)
            loss = criterion(conf_pred, tcp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(tuning_loader_confid):.6f}")

    end_time = time.time()
    training_time = end_time - start_time
    print("Training completed.")
    print(f"Total training time: {training_time:.2f} seconds")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "state_dict": confidnet.state_dict(),
            "training_time": training_time,
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
        },
        save_path,
    )


def evaluate_confidnet(seed: int) -> None:
    """
    Evaluate a trained ConfidNet model and save confidence scores for the test set.
    """
    ckpt_path = os.path.join(MODELS_DIR, f"confidnet_cifar10_resnet_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"No trained model found at: {ckpt_path}")

    layers = ARCHI_LAYERS.get('resnet')

    test_loader_confid, input_dim = get_confidnet_loader(
        feature_path=f"{RESULTS_DIR}/features_cifar10_resnet/seed{seed}/eval_features.pt",
        layer_key=layers[-2],
        train=False,
    )

    confidnet = get_confid_net_model(input_dim)
    confidnet.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["state_dict"])
    confidnet.to(DEVICE)
    confidnet.eval()

    all_confidences = []
    with torch.no_grad():
        for features, _ in test_loader_confid:
            features = features.to(DEVICE)
            out = confidnet(features)
            all_confidences.append(out.cpu().numpy())

    all_confidences = np.concatenate(all_confidences).flatten()

    save_path_confid = f"{RESULTS_DIR}/confidnet_cifar10_resnet_seed{seed}.npz"
    np.savez_compressed(save_path_confid, confidences=all_confidences)
    print(f"ConfidNet scores saved to: {save_path_confid}")