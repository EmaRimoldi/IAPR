from torchvision.transforms.v2 import functional as Fv2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torchvision.ops import masks_to_boxes
from torchvision.tv_tensors import BoundingBoxes
import torchvision.transforms.v2 as v2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from os import path, listdir, getcwd
from tqdm import tqdm

# Example of dataset definition 

# resize_transform = v2.Compose([
#     v2.Resize((IMG_SIZE, IMG_SIZE)),
#     v2.SanitizeBoundingBoxes(),
#     v2.ToImage(),
# ])

# train_dataset = DatasetAugmentation(
#     training_path="augmented_data/train", 
#     split_images=False,
#     perform_transformations=True
# )
# train_dataset.transforms = resize_transform


# Define a custom PyTorch Dataset class for dataset augmentation
class DatasetAugmentation(Dataset):
    def __init__(self, training_path, split_images=False, perform_transformations=True):
        # Store the root path of the training data
        self.training_path = training_path
        
        # Flag indicating if images/labels are split into subfolders or not
        self.split = split_images
        
        # Flag indicating whether to apply transformations or not
        self.transformation = perform_transformations
        
        # Determine the folder names for images and labels based on split_images flag
        self.image_dir = "split_images" if split_images else "images"
        self.label_dir = "split_labels" if split_images else "labels"

        # List and sort all image filenames in the image directory
        self.images = sorted(listdir(path.join(training_path, self.image_dir))) # crea una lista di immagini    
        
        # List and sort all label filenames in the label directory
        self.labels = sorted(listdir(path.join(training_path, self.label_dir)))

        # Compose a set of data augmentation transformations (with probabilities)
        self.transforms = v2.Compose([  # Dove applico le trasformazioni??
            v2.RandomResizedCrop(size=4000, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=(0.5, 1.5))], p=0.5),
            v2.RandomApply([v2.ColorJitter(contrast=(0.5, 1.5))], p=0.25),
            v2.RandomApply([v2.RandomRotation(360)], p=0.5),
            v2.RandomApply([v2.GaussianBlur(3)], p=0.3),
            v2.SanitizeBoundingBoxes()
        ])

        
    def __getitem__(self, idx):
        # Caricamento immagine e bounding boxes come prima
        img_path = path.join(self.training_path, self.image_dir, self.images[idx]) 
        label_path = path.join(self.training_path, self.label_dir, self.labels[idx])
        image = read_image(img_path, ImageReadMode.RGB)
        _, H, W = image.shape # otteniamo larghezza e altezza dell'immagine

        boxes_list = []
        labels_list = []
        with open(label_path, 'r') as f: # prenodiamo il file delle labels
            for line in f: #leggiamo riga per riga
                cls, cx, cy, w, h = map(float, line.strip().split()) # splittiamo la riga in 5 valori
                boxes_list.append([cx * W, cy * H, w * W, h * H]) # convertiamo in coordinate assolute
                labels_list.append(int(cls)) # convertiamo in int

        # Gestione immagini senza bounding boxes
        if len(boxes_list) == 0:
            # Crea una bounding box fittizia con dimensione nulla
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)

        boxes = BoundingBoxes(boxes, format=BoundingBoxFormat.CXCYWH, canvas_size=(H, W))

        sample = {
            "image": image,
            "bounding_boxes": boxes,
            "labels": labels
        }

        if self.transformation:
            sample = self.transforms(sample) # applica le trasformazioni all'immagine e alle bounding boxes

        return sample["image"], sample["bounding_boxes"], sample["labels"]



    def __len__(self):
        # Return the total number of images available
        return len(self.images)
