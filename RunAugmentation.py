################################################################################
# This script automates the process of preparing and augmenting an image dataset.
#
# It performs the following main tasks:
# 1. Splits a dataset of images and their corresponding label files into training
#    and validation subsets, based on a specified ratio.
#
# 2. Creates necessary folder structures for both the split datasets and the
#    augmented datasets, ensuring no duplicate directories are created.
#
# 3. Uses a DataAugmentation class to augment the training and validation data,
#    generating additional augmented images and labels to enrich the dataset.
#
# The code handles common image and label file formats (.png, .jpg, .jpeg, .txt),
# and ensures reproducibility by using a fixed random seed when splitting.
#
# Usage:
# - Place your original images and label files in the 'train/images' and
#   'train/labels' folders respectively.
# - Run this script to create split datasets and generate augmented data
#   inside the 'augmented_data' folder.
################################################################################

import os
import random
import shutil
from pathlib import Path
from DataAugmentation import DataAugmentation


def split_dataset(
    input_folder: str,
    output_train: str,
    output_val: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> None:
    """
    Split the dataset files from input_folder into training and validation sets,
    copying them into output_train and output_val folders respectively.

    Only files with extensions .png, .jpg, .jpeg, and .txt are considered.

    Parameters:
        input_folder (str): Path to the folder containing original files.
        output_train (str): Destination folder for training files.
        output_val (str): Destination folder for validation files.
        train_ratio (float): Ratio of data to be used for training.
        seed (int): Random seed for reproducibility.
    """

    # Create output directories only if they do not exist
    Path(output_train).mkdir(parents=True, exist_ok=True)
    Path(output_val).mkdir(parents=True, exist_ok=True)

    # List all relevant files in the input folder
    all_files = [f for f in os.listdir(input_folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', 'txt'))]
    all_files.sort()

    # Shuffle files with a fixed random seed
    random.seed(seed)
    random.shuffle(all_files)

    # Calculate split index
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # Copy training files
    for f in train_files:
        shutil.copy2(os.path.join(input_folder, f), os.path.join(output_train, f))

    # Copy validation files
    for f in val_files:
        shutil.copy2(os.path.join(input_folder, f), os.path.join(output_val, f))

    print(f"{len(train_files)} files copied to '{output_train}'")
    print(f"{len(val_files)} files copied to '{output_val}'")


def run_augmentation() -> None:
    """
    Runs the data augmentation pipeline:
    - Creates necessary folders for augmented data if not present.
    - Splits the original dataset into training and validation subsets.
    - Performs data augmentation on both subsets using the DataAugmentation class.
    """

    cwd = Path.cwd()

    # Base folder for augmented data
    augmented_data_path = cwd / "augmented_data"
    augmented_data_path.mkdir(exist_ok=True) #exist_ok=True creates the directory if it does not exist

    # Paths for train and validation augmented data
    train_path = augmented_data_path / "train"
    val_path = augmented_data_path / "val"
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    # Subfolders for images and labels inside train and val folders
    train_images_path = train_path / "images"
    train_labels_path = train_path / "labels"
    val_images_path = val_path / "images"
    val_labels_path = val_path / "labels"

    train_images_path.mkdir(exist_ok=True)
    train_labels_path.mkdir(exist_ok=True)
    val_images_path.mkdir(exist_ok=True)
    val_labels_path.mkdir(exist_ok=True)

    # Split original images and labels into train/val subsets
    split_dataset(
        input_folder=cwd / "train" / "images",
        output_train=cwd / "train" / "split" / "train" / "images",
        output_val=cwd / "train" / "split" / "val" / "images"
    )
    split_dataset(
        input_folder=cwd / "train" / "labels",
        output_train=cwd / "train" / "split" / "train" / "labels",
        output_val=cwd / "train" / "split" / "val" / "labels"
    )

    # Perform augmentation on training set
    DataAugmentation(
        nb=6,
        training_folder=str(cwd / 'train' / 'split' / 'train'),
        augmented_image_folder=str(train_images_path),
        augmented_label_folder=str(train_labels_path),
        add_base_images=True,
        split_images=False
    )

    # Perform augmentation on validation set
    DataAugmentation(
        nb=6,
        training_folder=str(cwd / 'train' / 'split' / 'val'),
        augmented_image_folder=str(val_images_path),
        augmented_label_folder=str(val_labels_path),
        add_base_images=True,
        split_images=False
    )


if __name__ == "__main__":
    run_augmentation()
