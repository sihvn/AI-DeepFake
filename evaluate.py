import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models


def validate(
    model: models.ResNet | models.EfficientNet,
    criterion: nn.CrossEntropyLoss,
    validate_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in validate_loader:
            # Move data to processor device (GPU if available)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(val_loss)
    return val_loss / len(validate_loader)


def evaluate(model: models.ResNet, data_loader: DataLoader, device: torch.device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            # Move data to processor device (GPU if available)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # move to CPU before converting to numpy
            predicted = predicted.cpu()
            labels = labels.cpu()

            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1
