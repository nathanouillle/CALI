import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 3, dropout_rate: float = 0.2) -> None:
        """
        Build a ResNet-18 backbone adapted for small images.

        Args:
            num_classes: Number of output classes.
            input_channels: Number of input image channels.
            dropout_rate: Dropout probability used in intermediate blocks.
        """
        super().__init__()
        self.dropout_rate = dropout_rate

        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = tv_models.resnet18(weights=weights)
        features_dim = 512

        # Adapt stem for small-resolution datasets (e.g., CIFAR).
        self.backbone.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        # Adapt classifier head.
        self.backbone.fc = nn.Linear(features_dim, num_classes)

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        """Run a forward pass with optional MC Dropout in evaluation mode."""
        use_dropout = mc_dropout or self.training

        def maybe_dropout(tensor: torch.Tensor) -> torch.Tensor:
            if use_dropout:
                return F.dropout(tensor, p=self.dropout_rate, training=True)
            return tensor
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = maybe_dropout(x)
        x = self.backbone.layer2(x)
        x = maybe_dropout(x)
        x = self.backbone.layer3(x)
        x = maybe_dropout(x)
        x = self.backbone.layer4(x)
        x = maybe_dropout(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = maybe_dropout(x)
        logits = self.backbone.fc(x)
        return logits
    
def create_model(num_classes: int = 10) -> nn.Module:
    """Create the default ResNet model configuration for CIFAR-like RGB data."""
    input_channels = 3
    dropout_rate = 0.2
    return ResNet(num_classes=num_classes, input_channels=input_channels, dropout_rate=dropout_rate)