"""Dataset metadata and DataLoader helpers."""

import json
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_selected_metadata(json_path):
    """Load selected iNaturalist JSON metadata into pandas DataFrames."""

    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = pd.DataFrame(data["categories"]).rename(columns={"id": "category_id"})
    images = pd.DataFrame(data["images"])
    annotations = pd.DataFrame(data["annotations"])

    return data, categories, images, annotations


def count_images_by_class(root_dir):
    """Count image files inside each class folder."""

    root_dir = Path(root_dir)
    rows = []

    for class_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        image_count = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        rows.append({"class_name": class_dir.name, "image_count": image_count})

    return pd.DataFrame(rows)


def build_transforms(image_size):
    """Create training and validation image transforms."""

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def create_dataloaders(train_dir, val_dir, image_size=224, batch_size=32, num_workers=0):
    """Create ImageFolder datasets and DataLoaders."""

    train_transform, val_transform = build_transforms(image_size)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataset, val_dataset, train_loader, val_loader
