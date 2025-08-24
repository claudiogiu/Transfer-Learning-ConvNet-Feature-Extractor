import os
import random
import torch
import torch.nn as nn
import numpy as np
import warnings
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from efficientnet_pytorch import EfficientNet
from augmentations import ImageAugmentations
from typing import List, Tuple

warnings.filterwarnings("ignore")

@dataclass
class ModelEvaluator:
    """
    Interface for evaluating a trained EfficientNet-B0 model on a test dataset.

    Attributes:
        weights_path (str): Path to the trained model weights (.pth file).
        num_classes (int): Number of output classes for classification.
        random_seed (int): Seed value for reproducibility across libraries.
        device (torch.device): Computation device selected automatically (CPU or GPU).
        augmentations (ImageAugmentations): Transformation pipeline applied to test data.
        data_dir (str): Automatically resolved path to the test dataset.
        test_loader (DataLoader): DataLoader instance for the test set.
        class_names (List[str]): List of class labels inferred from the dataset.
        model (nn.Module): EfficientNet-B0 model instance with a custom classification head.

    Methods:
        __post_init__() -> None:
            Initializes device, augmentation strategy, dataset path, model, and seed after instantiation.

        _set_seed(seed: int) -> None:
            Sets global random seed for reproducibility.

        _prepare_data() -> Tuple[DataLoader, List[str]]:
            Loads the test dataset and applies test-time transformations.

        _load_model() -> nn.Module:
            Reconstructs EfficientNet-B0 and loads trained weights from the 'models/' directory.
            Raises FileNotFoundError if the specified file does not exist.

        evaluate() -> None:
            Runs inference on the test set and prints confusion matrix, accuracy, and per-class performance.
    """

    weights_path: str
    num_classes: int = 5
    random_seed: int = 42

    device: torch.device = field(init=False)
    augmentations: ImageAugmentations = field(init=False)
    data_dir: str = field(init=False, default='data/processed')
    test_loader: DataLoader = field(init=False)
    class_names: List[str] = field(init=False)
    model: nn.Module = field(init=False)

    def __post_init__(self) -> None:
        self._set_seed(self.random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmentations = ImageAugmentations()
        self.test_loader, self.class_names = self._prepare_data()
        self.model = self._load_model()

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _prepare_data(self) -> Tuple[DataLoader, List[str]]:
        test_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'test'),
            transform=self.augmentations.test_transforms()
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        class_names = test_dataset.classes
        return test_loader, class_names

    def _load_model(self) -> nn.Module:
        model_path = os.path.join('models', self.weights_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        model = EfficientNet.from_pretrained('efficientnet-b0')
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, self.num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def evaluate(self) -> None:
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print("Model Evaluation Metrics:")
        print("\nConfusion Matrix:\n")
        print(cm)
        print(f"\nAccuracy: {acc * 100:.2f}%")
        print(f"Precision (macro): {precision * 100:.2f}%")
        print(f"Recall (macro): {recall * 100:.2f}%")
        print(f"F1 Score (macro): {f1 * 100:.2f}%")


if __name__ == "__main__":
    evaluator: ModelEvaluator = ModelEvaluator(
        weights_path='efficientnet_b0_transfer.pth',
        num_classes=5,
        random_seed=42
    )
    evaluator.evaluate()