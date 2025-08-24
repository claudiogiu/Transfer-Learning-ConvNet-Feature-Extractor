import os
import shutil
import random
import warnings
from dataclasses import dataclass
from typing import List, Optional

warnings.filterwarnings("ignore")

@dataclass
class ImageFolderSplitter:
    """
    Interface for splitting an image classification dataset into training and test sets.

    Attributes:
        dataset_name (str): Name of the folder inside 'data/raw' containing class subfolders.
        split_ratio (float): Proportion of images to allocate to the test set.
        random_seed (Optional[int]): Seed for random operations to ensure reproducibility.

    Properties:
        project_root (str): Absolute path to the project root directory.
        raw_path (str): Path to the raw dataset directory.
        processed_path (str): Path to the processed dataset directory.
        train_path (str): Destination path for training images.
        test_path (str): Destination path for test images.

    Methods:
        _prepare_directories() -> None:
            Creates the 'train' and 'test' directories under 'data/processed', replacing any existing ones.

        _split_dataset() -> None:
            Iterates through each class folder, shuffles and splits images into training and test sets,
            then copies them into the appropriate subdirectories.

        execute() -> None:
            Runs the preprocessing pipeline.
    """

    dataset_name: str
    split_ratio: float = 0.2
    random_seed: Optional[int] = 42 

    @property
    def project_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

    @property
    def raw_path(self) -> str:
        return os.path.join(self.project_root, "data", "raw", self.dataset_name)

    @property
    def processed_path(self) -> str:
        return os.path.join(self.project_root, "data", "processed")

    @property
    def train_path(self) -> str:
        return os.path.join(self.processed_path, "train")

    @property
    def test_path(self) -> str:
        return os.path.join(self.processed_path, "test")

    def _prepare_directories(self) -> None:
        for path in [self.train_path, self.test_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

    def _split_dataset(self) -> None:  
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Dataset folder not found: {self.raw_path}")

        self._prepare_directories()

        if self.random_seed is not None:
            random.seed(self.random_seed)

        class_folders: List[str] = [
            d for d in os.listdir(self.raw_path)
            if os.path.isdir(os.path.join(self.raw_path, d))
        ]
        if not class_folders:
            raise ValueError(f"No class folders found in {self.raw_path}")

        print(f"Dataset name: {self.dataset_name}")
        print(f"Total number of class folders: {len(class_folders)}")

        total_images = 0
        for class_name in class_folders:
            class_dir = os.path.join(self.raw_path, class_name)
            all_files = os.listdir(class_dir)
            images = [
                img for img in all_files
                if os.path.isfile(os.path.join(class_dir, img)) and img.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            total_images += len(images)

        print(f"Total number of valid image files: {total_images}")
        print("Starting train/test split for each class...")

        for class_name in class_folders:
            class_dir: str = os.path.join(self.raw_path, class_name)
            all_files: List[str] = os.listdir(class_dir)
            images: List[str] = [
                img for img in all_files
                if os.path.isfile(os.path.join(class_dir, img)) and img.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            if not images:
                print(f"No images found in class '{class_name}' — skipping.")
                continue

            random.shuffle(images)
            split_idx: int = int(len(images) * (1 - self.split_ratio))
            train_imgs: List[str] = images[:split_idx]
            test_imgs: List[str] = images[split_idx:]

            for subset_path, img_list in zip([self.train_path, self.test_path], [train_imgs, test_imgs]):
                subset_class_dir: str = os.path.join(subset_path, class_name)
                os.makedirs(subset_class_dir, exist_ok=True)
                for img_name in img_list:
                    src: str = os.path.join(class_dir, img_name)
                    dst: str = os.path.join(subset_class_dir, img_name)
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"Failed to copy {img_name} from {class_name}: {e}")

            print(f"[{class_name}] Train: {len(train_imgs)} | Test: {len(test_imgs)}")

    def execute(self) -> None:     
        self._split_dataset()


if __name__ == "__main__":
    dataset: str = "Grapevine_Leaves_Image_Dataset"
    processor: ImageFolderSplitter = ImageFolderSplitter(dataset_name=dataset, random_seed=42)
    processor.execute()