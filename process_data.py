import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


# ----------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------
class DeepfakeDetectionDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose = None):

        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Load data paths and labels
        for label, subdir in enumerate(["real_faces", "fake_faces"]):
            dir_path = os.path.join(self.root_dir, subdir)

            files = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            labels = [label] * len(files)  # 0 for real, 1 for fake

            self.data.extend(zip(files, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert(
            "RGB"
        )  # Convert to RGB to ensure 3 channels

        if self.transform:
            image = self.transform(image)

        return image, label


def inspect_dataloader(dataloader: DataLoader, device: torch.device) -> None:
    # Get the first batch
    images, labels = next(iter(dataloader))

    images = images.to(device)
    labels = labels.to(device)

    print("Is images cuda:", images.is_cuda)
    print("Is labels cuda:", labels.is_cuda)

    # Print the shapes and labels of the batch
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")


# Return train_loader, val_loader, test_loader
def get_data_loaders(
    dataset_root_dir: str,
    device: torch.device,
    seed: int = 33,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = DeepfakeDetectionDataset(dataset_root_dir, transform)

    # Split sizes
    total_size = len(dataset)
    # print(total_size)
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    val_size = total_size - train_size - test_size  # Remainder for validation

    print(train_size)
    print(val_size)
    print(test_size)

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(seed)
    )

    # DataLoaders
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    print("Inspecting Training DataLoader:")
    inspect_dataloader(train_loader, device)

    print("\nInspecting Validation DataLoader:")
    inspect_dataloader(val_loader, device)

    print("\nInspecting Testing DataLoader:")
    inspect_dataloader(test_loader, device)

    return train_loader, val_loader, test_loader
