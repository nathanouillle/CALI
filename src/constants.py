import torch

# PATHS
DATA_DIR = "./data"
RESULTS_DIR = "./results"
MODELS_DIR = "./models"

ARCHI_LAYERS = {
    "resnet": ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'logits'],
}

# MODELS
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CWAE default configuration
CWAE_CONFIG = {
    "latent_dim": 10,
    "epochs": 40,
    "batch_size": 128,
    "lr": 1e-3,
    "visualise": False,
    "lambda_cw": 10,
    "gamma_metric": 0.1,
    "margin_factor": 2.5,
}