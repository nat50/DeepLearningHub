from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATA_ROOT = "data"
FOOD101_RELATIVE_ROOT = Path("food-101")


class Food101Dataset(Dataset):
    """
    Food101 image classification dataset.
    """

    def __init__(self, transform=None):
        self.dataset_root = Path(DATA_ROOT).expanduser() / FOOD101_RELATIVE_ROOT
        self.image_root = self.dataset_root / "images"
        self.transform = transform

        folder_names = [path.name for path in self.image_root.iterdir() if path.is_dir()]
        self.classes = sorted(folder_names)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        self.samples = self._build_samples()
        self.targets = [label for _, label in self.samples]

    def _build_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []

        for class_name in self.classes:
            class_dir = self.image_root / class_name
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((str(image_path), self.class_to_idx[class_name]))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def make_stratified_split_indices(
    labels: Sequence[int],
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Dict[str, List[int]]:
    """
    Create deterministic stratified train/val/test splits for Food101.
    """

    indices = np.arange(len(labels))
    labels_arr = np.asarray(labels)

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_size,
        random_state=42,
        stratify=labels_arr,
    )

    val_ratio = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_ratio,
        random_state=42,
        stratify=labels_arr[temp_idx],
    )

    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def build_transforms(
    use_augmentation: bool = True,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Build a unified image classification pipeline for all backbones.
    """

    if use_augmentation:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
    use_augmentation: bool = True,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    drop_last_train: bool = False,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    device: torch.device | None = None,
):
    """
    Build Food101 train/val/test dataloaders using a stratified split.
    """

    train_tf = build_transforms(use_augmentation=use_augmentation, mean=mean, std=std)
    eval_tf = build_transforms(use_augmentation=False, mean=mean, std=std)
    eval_dataset = Food101Dataset(transform=eval_tf)
    train_dataset = Food101Dataset(transform=train_tf)

    split_indices = make_stratified_split_indices(
        eval_dataset.targets,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    train_subset = Subset(train_dataset, split_indices["train"])
    val_subset = Subset(eval_dataset, split_indices["val"])
    test_subset = Subset(eval_dataset, split_indices["test"])
    pin_memory = device.type == "cuda" if device is not None else torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_to_idx": eval_dataset.class_to_idx,
        "idx_to_class": eval_dataset.idx_to_class,
        "class_names": [eval_dataset.idx_to_class[idx] for idx in range(len(eval_dataset.idx_to_class))],
        "train_mode": "aug" if use_augmentation else "no_aug",
        "num_classes": len(eval_dataset.classes),
        "split_sizes": {split_name: len(indices) for split_name, indices in split_indices.items()},
        "dataset_root": eval_dataset.dataset_root,
    }


if __name__ == "__main__":
    loaders = create_dataloaders(
        batch_size=8,
        num_workers=0,
    )

    print("Num classes:", loaders["num_classes"])
    print("Split sizes:", loaders["split_sizes"])

    train_images, train_labels = next(iter(loaders["train_loader"]))
    val_images, val_labels = next(iter(loaders["val_loader"]))
    test_images, test_labels = next(iter(loaders["test_loader"]))

    print("Train batch:", tuple(train_images.shape), tuple(train_labels.shape))
    print("Val batch:", tuple(val_images.shape), tuple(val_labels.shape))
    print("Test batch:", tuple(test_images.shape), tuple(test_labels.shape))