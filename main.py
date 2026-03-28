import os
import random
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# ------------------------------
# Set seeds for reproducibility
# ------------------------------
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------
# Data transforms
# ------------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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

# ------------------------------
# Dataset classes
# ------------------------------
class ImageDataset(Dataset):
    """Returns (image, label, image_path) for a list of (path, label)."""
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
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
    """Returns (img1, img2, label) where label=1 if same class, 0 otherwise."""
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
        if random.random() < 0.5:
            # Positive pair: same class, different image
            pos_indices = [i for i in self.class_to_indices[anchor_label] if i != idx]
            if not pos_indices:
                # fallback to negative if no other image in class
                return self._get_negative_pair(idx)
            pos_idx = random.choice(pos_indices)
            pair_path, _ = self.data_list[pos_idx]
            target = 1
        else:
            # Negative pair: different class
            neg_label = random.choice([l for l in self.class_to_indices.keys() if l != anchor_label])
            neg_idx = random.choice(self.class_to_indices[neg_label])
            pair_path, _ = self.data_list[neg_idx]
            target = 0

        img1 = Image.open(anchor_path).convert('RGB')
        img2 = Image.open(pair_path).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(target, dtype=torch.float32)

    def _get_negative_pair(self, idx):
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
    """Returns (anchor, positive, negative) triplets."""
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
        # Positive: same class, different image
        pos_candidates = [i for i in self.class_to_indices[anchor_label] if i != idx]
        if not pos_candidates:
            # fallback: use anchor as positive (will give zero loss)
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

# ------------------------------
# Embedding Network
# ------------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, backbone='resnet50', embed_dim=256):
        super().__init__()

        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Only resnet50 supported")

        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

# ------------------------------
# Loss Functions
# ------------------------------
def contrastive_loss(emb1, emb2, label, margin=1.0):
    dist = F.pairwise_distance(emb1, emb2, p=2)
    loss = label * dist.pow(2) + (1 - label) * torch.clamp(margin - dist, min=0).pow(2)
    return loss.mean()

def triplet_loss(anchor, positive, negative, margin=0.3):
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0)
    return loss.mean()

def hard_negative_mining(embeddings, labels, margin=0.3):
    """In batch hard negative mining."""
    N = embeddings.size(0)
    distances = torch.cdist(embeddings, embeddings, p=2)  # (N, N)

    hardest_positive = torch.zeros(N, device=embeddings.device)
    hardest_negative = torch.zeros(N, device=embeddings.device)

    for i in range(N):
        same_class_mask = (labels == labels[i]).nonzero(as_tuple=True)[0]
        diff_class_mask = (labels != labels[i]).nonzero(as_tuple=True)[0]

        if len(same_class_mask) > 1:
            pos_dist = distances[i, same_class_mask]
            # Exclude self
            pos_dist[same_class_mask == i] = -float('inf')
            hardest_positive[i] = pos_dist.max()
        else:
            hardest_positive[i] = float('inf')

        if len(diff_class_mask) > 0:
            neg_dist = distances[i, diff_class_mask]
            hardest_negative[i] = neg_dist.min()
        else:
            hardest_negative[i] = -float('inf')

    valid = (hardest_positive != float('inf')) & (hardest_negative != -float('inf'))
    if valid.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
    loss = torch.clamp(hardest_positive[valid] - hardest_negative[valid] + margin, min=0).mean()
    return loss

# ------------------------------
# Evaluation functions
# ------------------------------
def validate_embeddings(embeddings, labels):
    """Compute pairwise cosine similarity (not used in training)."""
    sim = embeddings @ embeddings.T
    return sim

def evaluate_recall(model, dataloader, device, k=[1,5], save_embeddings_path=None):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Cosine similarity (already normalized)
    sim = all_embeddings @ all_embeddings.T
    sim.fill_diagonal_(-float('inf'))

    recalls = {}
    for k_val in k:
        _, topk_idx = sim.topk(k_val, dim=1)
        correct = (all_labels[topk_idx] == all_labels.unsqueeze(1)).any(dim=1)
        recalls[f"Recall@{k_val}"] = correct.float().mean().item()

    if save_embeddings_path:
        torch.save({'embeddings': all_embeddings, 'labels': all_labels}, save_embeddings_path)

    return recalls
#-------TSNE-------------
def tsne_visualization(embeddings, labels, title, save_path):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=SEED)
    emb_2d = tsne.fit_transform(embeddings.numpy())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab20', s=10)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()



def plot_training_curves(exp_name, epochs_list, losses, recalls1, recalls5, output_dir):
    plt.figure(figsize=(15, 6))
    
    # ------------------ Loss Plot ------------------
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, losses, marker='o', color='b', label='Loss')
    plt.title(f'Loss: {exp_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # ------------------ Recall Plot ------------------
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, recalls1, marker='s', color='g', label='R@1')
    plt.plot(epochs_list, recalls5, marker='^', color='r', label='R@5')
    plt.title(f'Recall: {exp_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    
    # ------------------ Save Plot ------------------
    save_path = os.path.join(output_dir, 'plots')
    os.makedirs(save_path, exist_ok=True)
    
    plt.savefig(os.path.join(save_path, f'curves_{exp_name}.png'))
    plt.close()

def retrieval_visualization(query_images, retrieved_images, query_labels, retrieved_labels, save_path):
    n_queries = len(query_images)
    if n_queries == 0:
        return

    k = len(retrieved_images[0])

    plt.figure(figsize=(2*(k+1), 2*n_queries))

    for i in range(n_queries):
        # Query image
        plt.subplot(n_queries, k+1, i*(k+1) + 1)
        plt.imshow(query_images[i])
        plt.title(f"Q: {str(query_labels[i])}", fontsize=8)
        plt.axis('off')

        # Retrieved images
        for j in range(k):
            plt.subplot(n_queries, k+1, i*(k+1) + j + 2)
            plt.imshow(retrieved_images[i][j])
            plt.title(str(retrieved_labels[i][j]), fontsize=8)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to Caltech101 folder')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Where to save models, logs, plots')
    parser.add_argument('--experiment', type=str, choices=['contrastive', 'triplet_random', 'triplet_hard', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--inference', action='store_true', help='Run inference on a single image')
    parser.add_argument('--image_path', type=str, help='Path to image for inference')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- Load dataset and split ----------
    print("Loading dataset...")
    full_dataset = ImageFolder(root=args.data_path, transform=None)  # load raw paths
    labels = [label for _, label in full_dataset.imgs]
    # Stratified split into train, val, test (e.g., 70% train, 15% val, 15% test)
    train_idx, temp_idx = train_test_split(
        np.arange(len(full_dataset)), test_size=0.3, stratify=labels, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=SEED
    )

    train_list = [full_dataset.imgs[i] for i in train_idx]   # (path, label)
    val_list   = [full_dataset.imgs[i] for i in val_idx]
    test_list  = [full_dataset.imgs[i] for i in test_idx]

    print(f"Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")

    # ---------- Inference mode ----------
    if args.inference:
        model = EmbeddingNet(embed_dim=128).to(device)
        # Load a pre-trained model (choose which experiment's best model)
        model_path = os.path.join(args.output_dir, 'models', 'contrastive_best.pth')
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Please train first.")
            return
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        img = Image.open(args.image_path).convert('RGB')
        img_tensor = val_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img_tensor)
        print("Embedding shape:", emb.shape)
        print("First 10 values:", emb[0, :10].cpu().numpy())
        return

    # ---------- Prepare data loaders for each experiment ----------
    def get_loaders(exp_name):
        train_loader = val_loader = test_loader = test_img_loader = None
        if exp_name == 'contrastive':
            train_dataset = ContrastiveDataset(train_list, transform=train_transform)
            val_dataset   = ContrastiveDataset(val_list, transform=val_transform)
            test_dataset  = ContrastiveDataset(test_list, transform=val_transform)

            # For evaluation, we also need image+label loaders
            test_img_dataset = ImageDataset(test_list, transform=val_transform)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            test_img_loader = DataLoader(test_img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            return train_loader, val_loader, test_loader, test_img_loader

        elif exp_name == 'triplet_random':
            train_dataset = TripletDataset(train_list, transform=train_transform)
            val_dataset   = TripletDataset(val_list, transform=val_transform)
            test_dataset  = TripletDataset(test_list, transform=val_transform)
            test_img_dataset = ImageDataset(test_list, transform=val_transform)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            test_img_loader = DataLoader(test_img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            return train_loader, val_loader, test_loader, test_img_loader

        elif exp_name == 'triplet_hard':
            # For hard mining we need standard image+label loaders
            train_img_dataset = ImageDataset(train_list, transform=train_transform)
            val_img_dataset   = ImageDataset(val_list, transform=val_transform)
            test_img_dataset  = ImageDataset(test_list, transform=val_transform)

            train_loader = DataLoader(train_img_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            val_loader   = DataLoader(val_img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            test_loader  = DataLoader(test_img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            test_img_loader = DataLoader(test_img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            return train_loader, val_loader, test_loader, test_img_loader
        else:
            raise ValueError(f"Unknown experiment: {exp_name}")
        return train_loader, val_loader, test_loader, test_img_loader

    # ---------- Run experiments ----------
    experiments = ['contrastive', 'triplet_random', 'triplet_hard'] if args.experiment == 'all' else [args.experiment]

    for exp in experiments:
        epoch_losses, epoch_r1, epoch_r5, epochs_list = [], [], [], []
        print(f"\n===== Starting experiment: {exp} =====")
        train_loader, val_loader, test_loader, test_img_loader = get_loaders(exp)

        model = EmbeddingNet(embed_dim=128).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_recall = 0.0
        csv_path = os.path.join(args.output_dir, f'training_log_{exp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'Recall1', 'Recall5'])

        for epoch in range(1, args.epochs+1):
            model.train()
            total_loss = 0.0

            if exp == 'contrastive':
                for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
                    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                    emb1 = model(img1)
                    emb2 = model(img2)
                    loss = contrastive_loss(emb1, emb2, labels, margin=1.0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            elif exp == 'triplet_random':
                for anchor, pos, neg in tqdm(train_loader, desc=f"Epoch {epoch}"):
                    anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                    emb_a = model(anchor)
                    emb_p = model(pos)
                    emb_n = model(neg)
                    loss = triplet_loss(emb_a, emb_p, emb_n, margin=0.3)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            elif exp == 'triplet_hard':
                for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
                    images, labels = images.to(device), labels.to(device)
                    embeddings = model(images)
                    loss = hard_negative_mining(embeddings, labels, margin=0.3)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Evaluate on test set
            recalls = evaluate_recall(model, test_img_loader, device, k=[1,5])
            r1 = recalls['Recall@1']
            r5 = recalls['Recall@5']

            # ----------- Store Training Metrics -----------
            epochs_list.append(epoch)
            epoch_losses.append(avg_loss)
            epoch_r1.append(r1)
            epoch_r5.append(r5)

            # Log to CSV
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ epoch, avg_loss, r1 , r5])
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Recall@1 = {r1:.4f}, Recall@5 = {r5:.4f}")

            # Save best model
            if r1 > best_recall:
                best_recall = r1
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'models', f'{exp}_best.pth'))
                torch.save(model.state_dict(), "/content/drive/MyDrive/Bilal Bushra_MSDS25051_03/model.pth") 
                print(f"Best model saved with Recall@1 = {best_recall:.4f}")

        # After training, final evaluation with best model
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'models', f'{exp}_best.pth')))
        final_recalls = evaluate_recall(model, test_img_loader, device, k=[1,5],
                                        save_embeddings_path=os.path.join(args.output_dir, f'embeddings_{exp}.pt'))
        print(f"Final Test Recall: {final_recalls}")

        # t-SNE
        emb_data = torch.load(os.path.join(args.output_dir, f'embeddings_{exp}.pt'))
        emb = emb_data['embeddings']
        lbl = emb_data['labels']
        if len(emb) > 2000:
            idx = np.random.choice(len(emb), 2000, replace=False)
            emb = emb[idx]
            lbl = lbl[idx]
        tsne_visualization(emb, lbl, f't-SNE: {exp}',
                           os.path.join(args.output_dir, 'plots', f'tsne_{exp}.png'))

        # Retrieval visualization (requires images)
        # We'll use test_img_loader (already has images)
                # Retrieval visualization
        model.eval()
        all_emb = []
        all_paths = []
        all_labels = []
        with torch.no_grad():
            for images, labels, paths in test_img_loader:
                images = images.to(device)
                emb = model(images)
                all_emb.append(emb.cpu())
                all_labels.append(labels)
                all_paths.extend(paths)
        all_emb = torch.cat(all_emb, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Similarity on CPU (safe)
        sim = (all_emb @ all_emb.T).cpu()

        # Select queries
        num_queries = min(10, len(all_emb))
        query_indices = np.random.choice(len(all_emb), num_queries, replace=False)

        query_images = []
        query_labels = []
        retrieved_images = []
        retrieved_labels = []

        for qi in query_indices:
            sim_row = sim[qi].clone()
            sim_row[qi] = -float('inf')

            top5_idx = sim_row.topk(5).indices

            # Query
            q_label_idx = all_labels[qi].item()
            query_images.append(Image.open(all_paths[qi]).convert('RGB'))
            query_labels.append(full_dataset.classes[q_label_idx])

            # Retrieved
            ret_imgs = []
            ret_lbls = []
            for ti in top5_idx:
                ret_imgs.append(Image.open(all_paths[ti]).convert('RGB'))
                t_label_idx = all_labels[ti].item()
                ret_lbls.append(full_dataset.classes[t_label_idx])

            retrieved_images.append(ret_imgs)
            retrieved_labels.append(ret_lbls)

        plot_training_curves(
            exp_name=exp,
            epochs_list=epochs_list,          
            losses=epoch_losses,
            recalls1=epoch_r1,
            recalls5=epoch_r5,
            output_dir=args.output_dir
        )
        # Visualization
        retrieval_visualization(
            query_images,
            retrieved_images,
            query_labels,
            retrieved_labels,
            os.path.join(args.output_dir, 'plots', f'retrieval_{exp}.png')
        )
        
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()