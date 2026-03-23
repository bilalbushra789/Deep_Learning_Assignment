import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from model import EmbeddingNet
from dataset import (
    ContrastiveDataset, TripletDataset, 
    ImageDataset, train_transform, val_transform
)
from loss import ContrastiveLoss, TripletLoss, batch_hard_mining
from retrieval import recall_at_k
import matplotlib.pyplot as plt

def save_checkpoint(state, filename):
    torch.save(state, filename)

def train_epoch(model, loader, optimizer, loss_fn, device, loss_type):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if loss_type == 'contrastive':
            img1, img2, target = batch
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            optimizer.zero_grad()
            emb1 = model(img1)
            emb2 = model(img2)
            loss = loss_fn(emb1, emb2, target)
        elif loss_type == 'triplet_random':
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            loss = loss_fn(emb_a, emb_p, emb_n)
        else:  # triplet_hard
            # batch is (images, labels)
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(images)
            loss = batch_hard_mining(embeddings, labels, margin=0.2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (images.size(0) if loss_type=='triplet_hard' else (img1.size(0) if loss_type=='contrastive' else anchor.size(0)))
    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    all_embs = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            imgs = imgs.to(device)
            embs = model(imgs)
            all_embs.append(embs.cpu())
            all_labels.append(lbls)
    all_embs = torch.cat(all_embs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    recall1 = recall_at_k(all_embs, all_labels, k=1)
    recall5 = recall_at_k(all_embs, all_labels, k=5)
    return recall1, recall5

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset splits (we assume you have prepared train_list, val_list, test_list as .npy files)
    train_list = np.load(args.train_list, allow_pickle=True)
    val_list = np.load(args.val_list, allow_pickle=True)
    test_list = np.load(args.test_list, allow_pickle=True)

    # Prepare data loaders based on loss type
    if args.loss == 'contrastive':
        train_dataset = ContrastiveDataset(train_list, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        loss_fn = ContrastiveLoss(margin=1.0)
    elif args.loss == 'triplet_random':
        train_dataset = TripletDataset(train_list, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        loss_fn = TripletLoss(margin=0.2)
    elif args.loss == 'triplet_hard':
        # For hard mining, we use a standard ImageDataset that returns (img, label)
        train_dataset = ImageDataset(train_list, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        loss_fn = None   # not used directly
    else:
        raise ValueError("Invalid loss type")

    # Validation and test datasets (always ImageDataset)
    val_dataset = ImageDataset(val_list, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dataset = ImageDataset(test_list, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = EmbeddingNet(embedding_dim=128).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_recall1 = 0.0
    patience_counter = 0
    train_losses = []
    val_recalls = []

    for epoch in range(1, args.epochs+1):
        # Train
        if args.loss == 'triplet_hard':
            loss = train_epoch(model, train_loader, optimizer, None, device, 'triplet_hard')
        else:
            loss = train_epoch(model, train_loader, optimizer, loss_fn, device, args.loss)
        train_losses.append(loss)
        print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f}")

        # Validate every val_interval epochs
        if epoch % args.val_interval == 0:
            recall1, recall5 = validate(model, val_loader, device)
            val_recalls.append((epoch, recall1, recall5))
            print(f"Validation Recall@1: {recall1:.4f}, Recall@5: {recall5:.4f}")

            # Save best model
            if recall1 > best_val_recall1:
                best_val_recall1 = recall1
                patience_counter = 0
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall1': recall1,
                    'loss': loss,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print("Saved best model.")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping after {epoch} epochs.")
                    break

    # Load best model and evaluate on test set
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_recall1, test_recall5 = validate(model, test_loader, device)
    print(f"Test Recall@1: {test_recall1:.4f}, Recall@5: {test_recall5:.4f}")

    # Save training curves
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {args.loss}')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f'train_loss_{args.loss}.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, required=True, choices=['contrastive', 'triplet_random', 'triplet_hard'])
    parser.add_argument('--train_list', type=str, required=True, help='Path to train list .npy')
    parser.add_argument('--val_list', type=str, required=True, help='Path to val list .npy')
    parser.add_argument('--test_list', type=str, required=True, help='Path to test list .npy')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='weights')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    main(args)