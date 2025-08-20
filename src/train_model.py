import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import DataLoader
from torchvision import datasets
from efficientnet_pytorch import EfficientNet
from augmentations import ImageAugmentations

warnings.filterwarnings("ignore")

@dataclass
class ModelTrainer:
    """
    Interface for training a transfer learning model using EfficientNet-B0,
    structured for compatibility with PyTorch's training workflow.

    Attributes:
        num_classes (int): Number of output classes for classification.
        epochs (int): Total number of training epochs.
        lr (float): Learning rate used by the optimizer.
        random_seed (int): Seed value for reproducibility across libraries.
        device (torch.device): Computation device selected automatically (CPU or GPU).
        model (Optional[nn.Module]): EfficientNet model instance with a custom classification head.
        augmentations (ImageAugmentations): Augmentation pipeline applied to training data.
        data_dir (str): Automatically resolved path to the training dataset.

    Methods:
        __post_init() -> None:
            Initializes device, augmentation strategy, and dataset path after instantiation.

        _set_seed(seed: int) -> None:
            Sets global random seed for reproducibility.

        _get_device() -> torch.device:
            Detects and returns the available computation device.

        _prepare_data() -> DataLoader:
            Loads and augments training data.

        _build_model() -> nn.Module:
            Loads EfficientNet-B0 and replaces the final layer.

        train() -> None:
            Executes the training loop and monitors training progress.

        _save_model() -> None:
            Saves the trained model weights.
    """

    num_classes: int = 5
    epochs: int = 30
    lr: float = 0.001
    random_seed: int = 42
    device: torch.device = field(init=False)
    model: Optional[nn.Module] = field(init=False, default=None)
    augmentations: ImageAugmentations = field(init=False)
    data_dir: str = field(init=False)

    def __post_init__(self) -> None:
        self._set_seed(self.random_seed)
        self.device = self._get_device()
        self.augmentations = ImageAugmentations()
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_device(self) -> torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        return device

    def _prepare_data(self) -> DataLoader:
        train_path = os.path.join(self.data_dir, 'train')
        train_dataset = datasets.ImageFolder(
            train_path,
            transform=self.augmentations.training_transforms()
        )
        train_loader: DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_loader

    def _build_model(self) -> nn.Module:
        model: nn.Module = EfficientNet.from_pretrained('efficientnet-b0')
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs: int = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, self.num_classes)
        model.to(self.device)
        return model

    def train(self) -> None:
        train_loader: DataLoader = self._prepare_data()
        self.model = self._build_model()

        criterion: nn.Module = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = optim.Adam(self.model._fc.parameters(), lr=self.lr)

        print("Starting training...")
        for epoch in range(self.epochs):
            running_loss: float = 0.0
            correct: int = 0
            total: int = 0
            self.model.train()

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_loss: float = running_loss / len(train_loader)
            accuracy: float = correct / total
            print(f"Epoch {epoch+1}/{self.epochs} - Avg Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

        self._save_model()

    def _save_model(self) -> None:
        models_dir: str = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path: str = os.path.join(models_dir, 'efficientnet_b0_transfer.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    trainer: ModelTrainer = ModelTrainer(
        num_classes=5,
        epochs=30,
        lr=0.001,
        random_seed=42
    )
    trainer.train()