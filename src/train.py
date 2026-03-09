import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import create_model
from .utils import get_loader_cifar10, set_seed
from .constants import DEVICE, MODELS_DIR


os.makedirs(MODELS_DIR, exist_ok=True)


def train_resnet(
    model: nn.Module,
    train_loader: DataLoader,
    seed: int,
) -> None:
    """Train a model for one seed and save a checkpoint."""
    ckpt_path = os.path.join(MODELS_DIR, f"cifar10_resnet_seed{seed}.pt")
    if os.path.exists(ckpt_path):
        print(f"[SKIP TRAINING] cifar10 | seed {seed} - checkpoint already exists.")
        return

    if len(train_loader) == 0:
        raise ValueError("Training loader is empty.")

    criterion_train = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=4)

    model.train()
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion_train(outputs, y.view(-1).long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
    torch.save(model.state_dict(), ckpt_path)


def train_model(seed: int) -> None:
    """Train a model for a given seed and dataset."""
    set_seed(seed)
    train_loader, _, num_classes = get_loader_cifar10()
    model = create_model(num_classes=num_classes).to(DEVICE)
    train_resnet(model, train_loader, seed)
