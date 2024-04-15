import torch
import torch.nn as nn
from torchvision import models

ResNet_models = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
EfficientNet_models = [
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
]


def get_model(
    model_name: str, device: torch.device
) -> models.ResNet | models.EfficientNet:
    # Pre-trained ResNet models
    if model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "ResNet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == "ResNet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    elif model_name == "ResNet152":
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    # Pre-trained EfficientNet models
    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
    elif model_name == "EfficientNetB1":
        model = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2
        )
    elif model_name == "EfficientNetB2":
        model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
    elif model_name == "EfficientNetB3":
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
    elif model_name == "EfficientNetB4":
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
    elif model_name == "EfficientNetB5":
        model = models.efficientnet_b5(
            weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1
        )
    elif model_name == "EfficientNetB6":
        model = models.efficientnet_b6(
            weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1
        )
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Modify the final layer for binary classification
    if model_name in ResNet_models:
        # Get the number of input features for the last FC layer
        num_features = model.fc.in_features

        # Modify the last FC layer for binary classification
        model.fc = nn.Linear(num_features, 2)

        # # Modify the last FC layer for binary classification with dropout
        # model.fc = nn.Sequential(
        #     nn.Linear(num_features, 512),  # Adding an intermediate layer (optional)
        #     nn.ReLU(inplace=True),  # Adding activation function (optional)
        #     nn.Dropout(p=0.5),  # Dropout layer with dropout probability of 0.5
        #     nn.Linear(512, 2),  # Final output layer
        # )

    if model_name in EfficientNet_models:
        # Get the number of input features for the last FC layer
        num_features = model.classifier[1].in_features

        # Modify the last FC layer for binary classification
        model.classifier[1] = nn.Linear(num_features, 2)

        # # Modify the last FC layer for binary classification with dropout
        # model.classifier[1] = nn.Sequential(
        #     nn.Linear(num_features, 512),  # Adding an intermediate layer (optional)
        #     nn.ReLU(inplace=True),  # Adding activation function (optional)
        #     nn.Dropout(p=0.5),  # Dropout layer with dropout probability of 0.5
        #     nn.Linear(512, 2),  # Final output layer
        # )

    # Move the model to GPU if available
    model = model.to(device)

    return model
