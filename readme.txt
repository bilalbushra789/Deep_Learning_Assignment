================================================
README.txt – Deep Learning Assignment 
================================================

This code implements contrastive learning, triplet random sampling, and triplet hard mining for learning image embeddings using a ResNet-50 backbone. The models are trained on Caltech101 and evaluated with Recall@k and t‑SNE visualizations.

------------------------------------------------
Requirements
------------------------------------------------
- Python 3.8+
- PyTorch >= 1.12
- torchvision >= 0.13
- numpy, matplotlib, scikit-learn, tqdm, Pillow

Install dependencies:
    pip install torch torchvision numpy matplotlib scikit-learn tqdm pillow

------------------------------------------------
Dataset
------------------------------------------------
Place the Caltech101 dataset in a directory (e.g., /path/to/caltech101). The directory should contain subfolders named after each class, each containing images. The code uses torchvision.datasets.ImageFolder, so the structure is:

    caltech101/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            ...
        ...

------------------------------------------------
Running the Code
------------------------------------------------
Basic usage:
    python main.py --data_path /path/to/caltech101

This will run all three experiments (contrastive, triplet_random, triplet_hard) with default settings.

Command line arguments:
    --data_path      Path to the Caltech101 folder (required)
    --output_dir     Directory to save outputs (default: ./outputs)
    --experiment     Which experiment to run: 'contrastive', 'triplet_random', 'triplet_hard', or 'all' (default: all)
    --epochs         Number of training epochs (default: 20)
    --batch_size     Batch size (default: 32)
    --lr             Learning rate (default: 1e-4)
    --inference      If set, run inference on a single image (requires --image_path)
    --image_path     Path to an image for inference (used with --inference)

Examples:
    # Run only contrastive learning for 10 epochs
    python main.py --data_path ./caltech101 --experiment contrastive --epochs 10

    # Run all experiments with batch size 32
    python main.py --data_path ./caltech101 --batch_size 32

    # Run inference on a test image using the best contrastive model
    python main.py --data_path ./caltech101 --inference --image_path test.jpg

------------------------------------------------
Output Structure
------------------------------------------------
After running, the following files are created in the output_dir:

outputs/
├── models/                     # Saved best models
│   ├── contrastive_best.pth
│   ├── triplet_random_best.pth
│   └── triplet_hard_best.pth
├── plots/                      # Visualizations
│   ├── tsne_contrastive.png
│   ├── tsne_triplet_random.png
│   ├── tsne_triplet_hard.png
│   ├── retrieval_contrastive.png
│   ├── retrieval_triplet_random.png
│   └── retrieval_triplet_hard.png
├── training_log_*.csv          # Training logs with loss and Recall@1 per epoch
└── embeddings_*.pt             # Saved embeddings and labels for test set

------------------------------------------------
Evaluation
------------------------------------------------
Recall@1 and Recall@5 are computed on the test set after each epoch, and the best model (by Recall@1) is saved. After training, the final recall values are printed and embeddings are saved for t‑SNE.

The retrieval visualizations show 10 random query images and their top‑5 retrieved neighbors (green = correct class, red = wrong).

------------------------------------------------
Notes
------------------------------------------------
- The code uses a fixed random seed (7) for reproducibility.
- It automatically splits the dataset into 70% train, 15% validation, 15% test (stratified by class).
- For triplet hard mining, in‑batch hard negative mining is used.
- The embedding dimension is fixed to 256.
