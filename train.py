import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models


# ----------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------
def train_model(
    model: models.ResNet,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    dataloader: DataLoader,
    num_epochs: int,
) -> models.ResNet:
    # Assign GPU as device if available, else assign cpu
    print("Cuda is available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda device:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0

        for inputs, labels in dataloader:
            # print(f"Input batch shape: {inputs.shape}")
            # print(f"Labels batch shape: {labels.shape}")

            # Move data to cuda device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct_preds.double() / len(dataloader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
        )

    print("Training complete")
    return model
