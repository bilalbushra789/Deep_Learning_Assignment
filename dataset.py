import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# Standard ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transformation for training (with augmentation) and validation/test
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class ImageDataset(Dataset):
    
    #Simple dataset that returns (image, label, image_path) for a list of (path, label).
    #Used for evaluation and precomputation.
    
    def __init__(self, data_list, transform=None):
        self.data_list = data_list   # list of (img_path, label)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

class ContrastiveDataset(Dataset):
    
    #Returns pairs (img1, img2, label) where label=1 if same class, 0 otherwise.
    #Positive and negative pairs are sampled with equal probability (p=0.5).
    
    def __init__(self, data_list, transform=None):
        self.data_list = data_list   # list of (img_path, label)
        self.transform = transform
        # Build class-to-indices mapping
        self.class_to_indices = {}
        for idx, (_, lbl) in enumerate(data_list):
            self.class_to_indices.setdefault(lbl, []).append(idx)

    def __len__(self):
        return len(self.data_list)   # number of pairs = number of images (each image used as anchor once)

    def __getitem__(self, idx):
        # Anchor image info
        anchor_path, anchor_label = self.data_list[idx]
        # Decide positive (1) or negative (0) pair
        if random.random() < 0.5:
            # Positive pair: same class, different image
            pos_indices = [i for i in self.class_to_indices[anchor_label] if i != idx]
            if not pos_indices:
                # fallback to negative if no other image in class
                return self._get_negative_pair(idx)
            pos_idx = random.choice(pos_indices)
            pair_path, pair_label = self.data_list[pos_idx]
            target = 1
        else:
            # Negative pair: different class
            neg_label = random.choice([l for l in self.class_to_indices.keys() if l != anchor_label])
            neg_idx = random.choice(self.class_to_indices[neg_label])
            pair_path, pair_label = self.data_list[neg_idx]
            target = 0

        # Load images
        img1 = Image.open(anchor_path).convert('RGB')
        img2 = Image.open(pair_path).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(target, dtype=torch.float32)

    def _get_negative_pair(self, idx):
        """Fallback to return a negative pair."""
        anchor_path, anchor_label = self.data_list[idx]
        neg_label = random.choice([l for l in self.class_to_indices.keys() if l != anchor_label])
        neg_idx = random.choice(self.class_to_indices[neg_label])
        pair_path, _ = self.data_list[neg_idx]
        img1 = Image.open(anchor_path).convert('RGB')
        img2 = Image.open(pair_path).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(0.0, dtype=torch.float32)

class TripletDataset(Dataset):
    
    #Returns triplets (anchor, positive, negative) where positive is same class,
    #negative is different class.

    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.class_to_indices = {}
        for idx, (_, lbl) in enumerate(data_list):
            self.class_to_indices.setdefault(lbl, []).append(idx)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        anchor_path, anchor_label = self.data_list[idx]
        # Positive: same class, different index
        pos_candidates = [i for i in self.class_to_indices[anchor_label] if i != idx]
        if not pos_candidates:
            # If no positive available, return a dummy triplet? Should not happen if class size >1.
            # Fallback: use anchor as positive (will cause zero loss but avoid error)
            pos_idx = idx
        else:
            pos_idx = random.choice(pos_candidates)
        # Negative: different class
        neg_label = random.choice([l for l in self.class_to_indices.keys() if l != anchor_label])
        neg_idx = random.choice(self.class_to_indices[neg_label])

        img_a = Image.open(anchor_path).convert('RGB')
        img_p = Image.open(self.data_list[pos_idx][0]).convert('RGB')
        img_n = Image.open(self.data_list[neg_idx][0]).convert('RGB')
        if self.transform:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)
        return img_a, img_p, img_n