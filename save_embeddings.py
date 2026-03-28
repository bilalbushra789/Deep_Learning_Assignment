import torch
import numpy as np
from model import EmbeddingNet
from dataset import ImageDataset, val_transform
from torch.utils.data import DataLoader
import argparse
import os

def compute_embeddings(model, dataloader, device):
    #Return embeddings and labels as tensors.
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, lbls, _ in dataloader:
            imgs = imgs.to(device)
            emb = model(imgs)
            embeddings.append(emb.cpu())
            labels.append(lbls)
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

import torch.nn.functional as F

def compute_recall(embeddings, labels, k=5):
    """Compute Recall@K using cosine similarity."""
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim = embeddings @ embeddings.T
    sim.fill_diagonal_(-float('inf'))
    _, topk_idx = sim.topk(k, dim=1)
    correct = (labels[topk_idx] == labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = EmbeddingNet(embedding_dim=128).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Prepare dataset (assuming data_list is saved as .npy or we rebuild from split)
    # For simplicity, we assume you have a list of (path, label) saved as .npy
    data_list = np.load(args.data_list_file, allow_pickle=True)
    dataset = ImageDataset(data_list, transform=val_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    embeddings, labels = compute_embeddings(model, loader, device)
    # Save
    np.save(os.path.join(args.out_dir, 'embeddings.npy'), embeddings.numpy())
    np.save(os.path.join(args.out_dir, 'labels.npy'), labels.numpy())
    print(f"Saved embeddings and labels to {args.out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_list_file', type=str, required=True, help='Path to .npy file containing list of (path, label)')
    parser.add_argument('--out_dir', type=str, default='embeddings', help='Directory to save embeddings')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)