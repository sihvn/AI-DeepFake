import torch
import torch.nn as nn
from torchvision import models


def get_model() -> models.ResNet:
    # Assign GPU as device if available, else assign cpu
    print("Cuda is available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda device:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load a pre-trained ResNet50 model
    model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")

    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Output for 2 classes: fake and real

    model = model.to(device)

    return model
