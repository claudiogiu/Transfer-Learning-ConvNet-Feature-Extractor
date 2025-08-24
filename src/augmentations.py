from dataclasses import dataclass
from typing import Optional
from torchvision import transforms

@dataclass
class ImageAugmentations:
    """
    Interface for defining transformation strategies to enhance and prepare image data for classification tasks,
    structured for compatibility with PyTorch's ImageFolder.

    Attributes:
        crop_size (int): Final crop size applied to input images.
        mean (Optional[list]): Mean values for normalization.
        std (Optional[list]): Standard deviation values for normalization.

    Methods:
        training_transforms() -> transforms.Compose:
            Returns a randomized sequence of augmentations applied to training data.

        test_transforms() -> transforms.Compose:
            Returns a deterministic sequence of preprocessing steps applied to test data.
    """

    crop_size: int = 224
    mean: Optional[list] = None
    std: Optional[list] = None

    def __post_init__(self):
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]

    def training_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.RandomResizedCrop(self.crop_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=255
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def test_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])