import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models


# ----------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------
def train_model(
    model: models.ResNet | models.EfficientNet,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
) -> models.ResNet:
    print("Training in progress...")

    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0

        for inputs, labels in dataloader:
            # print(f"Input batch shape: {inputs.shape}")
            # print(f"Labels batch shape: {labels.shape}")

            # Move data to processor device (GPU if available)
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
            f"    Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
        )

    print("Training complete.\n")
    return model
