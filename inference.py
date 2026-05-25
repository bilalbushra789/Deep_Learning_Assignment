import torch
import argparse
from PIL import Image
from model import EmbeddingNet
from dataset import val_transform
import numpy as np

def load_model(checkpoint_path, device):
    model = EmbeddingNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('backbone.', 'backbone.resnet.') if 'backbone.' in k and 'resnet' not in k else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def infer_single_image(model, image_path, device):
    image = Image.open(image_path).convert('RGB')
    image = val_transform(image).unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        embedding = model(image)
    return embedding.cpu().numpy().squeeze()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)

    if args.image_path:
        emb = infer_single_image(model, args.image_path, device)
        print(f"Embedding shape: {emb.shape}")
        np.save(args.output, emb) if args.output else print(emb)
    elif args.image_list:
        # Process multiple images and save as a single .npy file
        embs = []
        with open(args.image_list, 'r') as f:
            paths = [line.strip() for line in f]
        for path in paths:
            emb = infer_single_image(model, path, device)
            embs.append(emb)
        embs = np.stack(embs)
        np.save(args.output, embs)
        print(f"Saved embeddings for {len(paths)} images to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image_path', type=str, help='Single image file')
    parser.add_argument('--image_list', type=str, help='Text file with image paths, one per line')
    parser.add_argument('--output', type=str, default='embedding.npy', help='Output .npy file')
    args = parser.parse_args()
    main(args)
    print("Inference completed.")