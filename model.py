import torch
import torch.nn as nn
from torchvision import models

ResNet_models = ["ResNet50"]
EfficientNet_models = ["EfficientNetB0"]


def get_model(
    model_name: str, device: torch.device
) -> models.ResNet | models.EfficientNet:
    # Load a pre-trained ResNet50 model

    if model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
    else:
        model = models.resnet50(weights="IMAGENET1K_V2")

    # Modify the final layer for binary classification
    if model_name in ResNet_models:
        # Get the number of input features for the last FC layer
        num_features = model.fc.in_features
        # Modify the last FC layer for binary classification
        model.fc = nn.Linear(num_features, 2)

    if model_name in EfficientNet_models:
        # Get the number of input features for the last FC layer
        num_features = model.classifier[1].in_features
        # Modify the last FC layer for binary classification
        model.classifier[1] = nn.Linear(num_features, 2)

    # Move the model to GPU if available
    model = model.to(device)

    return model
