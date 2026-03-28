import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
   
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, target):
        # emb1, emb2: (batch, emb_dim)
        # target: (batch) with values 0 or 1
        dist = F.pairwise_distance(emb1, emb2)  # (batch)
        loss = target * dist.pow(2) + (1 - target) * F.relu(self.margin - dist).pow(2)
        return loss.mean()

class TripletLoss(nn.Module):
   
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

def batch_hard_mining(embeddings, labels, margin=0.2):
   
    batch_size = embeddings.size(0)
    # Compute pairwise distance matrix
    dist_matrix = torch.cdist(embeddings, embeddings)  # (batch, batch)

    losses = []
    for i in range(batch_size):
        # Mask for same class (excluding anchor itself)
        same_class = (labels == labels[i]).float()
        same_class[i] = 0
        if same_class.sum() == 0:
            continue   # no positive in batch
        
        # Hardest positive: maximum distance among same class
        pos_dist = dist_matrix[i] [same_class.bool()].max()

        # Mask for different class
        diff_class = 1 - (labels == labels[i]).float()
        # To avoid selecting same class, set those distances to a large number
        neg_dist = (dist_matrix[i] + 1e6 * (1 - diff_class)).min()

        loss = F.relu(pos_dist - neg_dist + margin)
        losses.append(loss)
        # Fix: add large penalty to same class distances, then take min
        neg_dist = (dist_matrix[i] + 1e6 * (labels == labels[i]).float()).min()
        loss = F.relu(pos_dist - neg_dist + margin)
        losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    return torch.stack(losses).mean()
if __name__ == "__main__":
    print("Loss functions loaded successfully.")