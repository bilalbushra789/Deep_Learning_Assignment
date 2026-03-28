import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from PIL import Image

def recall_at_k(embeddings, labels, k=1):
    
    #Compute Recall@K for the given embeddings and labels.
    #embeddings: (num_samples, emb_dim) tensor
    #labels: (num_samples) tensor
    
    embeddings = F.normalize(embeddings, p=2, dim=1)  # ensure normalized
    dist_matrix = torch.cdist(embeddings, embeddings)  # (N, N)
    sorted_indices = dist_matrix.argsort(dim=1)        # nearest first

    correct = 0
    for i, label in enumerate(labels):
        topk = sorted_indices[i, 1:k+1]   # exclude self
        if any(labels[idx] == label for idx in topk):
            correct += 1
    return correct / len(labels)

def show_retrieval(query_idx, embeddings, images, labels, class_names, k=5, save_path=None):
    
    #Display query image and top-k retrieved images.
    # query_idx: index of query image in the dataset.
    
    embeddings = F.normalize(embeddings, p=2, dim=1)
    dist = torch.cdist(embeddings[query_idx:query_idx+1], embeddings).squeeze()
    # Exclude query itself
    nearest = dist.argsort()[1:k+1]   # indices of top k

    fig = plt.figure(figsize=(15, 3))
    gs = gridspec.GridSpec(1, k+1, width_ratios=[1]* (k+1))

    # Query
    ax = plt.subplot(gs[0])
    if isinstance(images[query_idx], torch.Tensor):
        img = images[query_idx].permute(1,2,0).numpy()
        img = img * np.array(std) + np.array(mean)   # unnormalize
        img = np.clip(img, 0, 1)
    else:
        img = images[query_idx]
    ax.imshow(img)
    ax.set_title("Query\n" + class_names[labels[query_idx]])
    ax.axis('off')

    # Neighbors
    for j, idx in enumerate(nearest):
        ax = plt.subplot(gs[j+1])
        if isinstance(images[idx], torch.Tensor):
            img = images[idx].permute(1,2,0).numpy()
            img = img * np.array(std) + np.array(mean)
            img = np.clip(img, 0, 1)
        else:
            img = images[idx]
        ax.imshow(img)
        correct = (labels[idx] == labels[query_idx])
        color = 'green' if correct else 'red'
        ax.set_title(class_names[labels[idx]], color=color)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_tsne(embeddings, labels, class_names=None, title='t-SNE', save_path=None):

    #Apply t-SNE to embeddings and produce a scatter plot colored by class.
    #embeddings: (N, emb_dim) numpy array or tensor.
    #labels: (N) list/array of labels.
    #class_names: optional dict for legend.
    
    from sklearn.manifold import TSNE
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    labels = np.array(labels)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        idx = labels == lbl
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=class_names[lbl] if class_names else str(lbl), s=5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    print("recall@1 =", recall_at_k(torch.randn(100,128), torch.randint(0,10,(100,)), k=1))
    print("t-SNE plot test: call plot_tsne(emb, lbl, names)")
    print("Retrieval test: call show_retrieval(0, emb, images, labels, class_names)")