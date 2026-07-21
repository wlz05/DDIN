# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

**DDIN** is a deep learning framework for **multimodal fake news detection**. It leverages a domain-aware disentanglement and interaction network to capture cross-modal inconsistencies between text and images, enabling robust identification of misinformation.

Designed for fake news detection on multiple multimodal datasets including Weibo, Weibo-21, and FineFake.

Paper Link: 

---

## Architecture

```
┌──────────────────────────────────────────────────────────
│                   DDIN Architecture                     
├──────────────────────────────────────────────────────────
│  (a) Dual-Stream Multi-Granularity Feature Extraction    
│      BERT (text local) + MAE (image local)               
│      + CLIP (text-image global)                          
│                         ↓                                
│  (b) Multi-Scale Semantic Projection                     
│                         ↓                                
│  (c) Multi-Granularity Cross-Modal Inconsistency Mining  
│      ├── Global-Global Inconsistency                     
│      ├── Global-Local Inconsistency     
│      └── Local-Local Inconsistency (Cross-Attention)                      
│                         ↓                                
│  (d) Hierarchical Conflict Synergy Network               
│                         ↓                                
│  (e) Domain-Adaptive Inconsistency Weighting              
│                         ↓                                
│  (f) Multimodal Global Fusion → Classifier               
└──────────────────────────────────────────────────────────
```

### Key Contributions

- **Multi-Granularity Inconsistency Mining**: Captures conflict signals between text and images at three granularities — Global-Global, Global-Local, and Local-Local.
- **Hierarchical Conflict Synergy**: A Transformer-based module that enables conflict features at different granularities to communicate and reinforce each other.
- **Domain-Adaptive Weighting**: Dynamically adjusts the importance of different inconsistency signals based on the news domain (9 categories: technology, military, education, etc.).
- **Multi-Scale Semantic Projection**: Employs multiple parallel projection channels to capture polysemous semantic correspondences between text and images.

---

## Project Structure

```
DDIN/
├── model/
│   ├── net.py                 # DDIN core model + Trainer
│   ├── layers.py              # Base layers (MLP, Attention, FocalLoss, etc.)
│   ├── pivot.py               # Hypergraph convolution
│   ├── bert.py                # BERT modules
│   ├── domain.py              # Multi-domain PLE-FEND model variant
│   ├── weibo.py               # Weibo domain model variant
│   ├── w21.py                 # Weibo21 domain model variant
│   ├── gossip.py              # GossipCop/FineFake model
│   ├── raw.py                 # Raw DDIN model variant
│   ├── clip.py                # CLIP domain module
│   └── test.py                # Test script
├── cnn/
│   ├── resnet.py              # ResNet
│   ├── vgg.py                 # VGG
│   ├── efficient.py           # EfficientNet
│   ├── inception.py           # InceptionNet
│   ├── lenet.py               # LeNet-5
│   ├── unet.py                # U-Net
│   ├── nn.py                  # Network modules
│   └── fp16.py                # Mixed precision utils
├── utils/
│   ├── loader.py              # Generic data loader
│   ├── clipld.py              # Weibo data loader
│   ├── wloader.py             # Weibo CLIP image loader
│   ├── w21ld.py               # Weibo21 data loader
│   ├── fld.py                 # FineFake data loader
│   ├── utils.py               # Metrics, Recorder, clipdata2gpu, data2gpu
│   ├── extract.py             # FineFake CLIP feature extraction
│   ├── fsplit.py              # Fast data split
│   ├── fiximg.py              # Image fix utils
│   ├── datasets.py            # Dataset processing
│   ├── crop.py                # Image cropping
│   ├── lars.py                # LARS optimizer
│   ├── decay.py               # LR decay
│   ├── sched.py               # LR scheduling
│   ├── misc.py                # Miscellaneous utils
│   └── pos.py                 # Positional encoding
├── data/                      # Weibo dataset (CSV + generated pkl)
│   ├── train_origin.csv
│   ├── val_origin.csv
│   ├── test_origin.csv
│   ├── nonrumor_images/
│   └── rumor_images/
├── weibo21/                   # Weibo21 dataset (Excel + images + generated pkl)
│   ├── train_datasets.xlsx
│   ├── val_datasets.xlsx
│   ├── test_datasets.xlsx
│   ├── nonrumor_images/
│   └── rumor_images/
├── FineFake/                  # FineFake dataset
│   ├── FineFake.pkl
│   ├── gossip_train.csv
│   ├── gossip_test.csv
│   ├── gossip_train/
│   ├── f_train_loader.pkl
│   └── f_train_clip.pkl
├── w21/                       # Weibo21 data processing scripts
│   ├── data.py
│   ├── data2.py
│   ├── probe.py
│   └── config.py
├── main.py                    # Entry point (argparse + config)
├── run.py                     # Training dispatch (3 datasets, DDIN + Gossip models)
├── mae.py                     # MAE ViT model
├── dataset.py                 # FineFake/GossipCop dataset
├── feature.py                 # t-SNE feature visualization
├── preproc.py                 # Weibo MAE image preprocessing -> data/
├── clipprep.py                # Weibo CLIP image preprocessing -> data/
├── w21prep.py                 # Weibo21 MAE image preprocessing -> weibo21/
├── w21clip.py                 # Weibo21 CLIP image preprocessing -> weibo21/
├── split.py                   # Reasoning column split utility
├── probe.py                   # Test probe
├── requirements.txt           # Python dependencies
└── .gitignore
```

---

## Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.10 (Ubuntu 22.04) |
| PyTorch | 2.1.0 |
| CUDA | 12.1 |
| Transformers | latest |
| cn_clip | latest |
| timm | latest |

### Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch==2.1.0` — Deep learning framework
- `transformers` — Pre-trained models (BERT, etc.)
- `cn_clip` — Chinese CLIP model
- `openai/CLIP` — OpenAI CLIP model
- `timm` — Vision Transformer and model components
- `positional_encodings` — Positional encoding utilities
- `scikit-learn` — Machine learning utilities
- `pandas`, `openpyxl` — Data processing

---

## Pretrained Models

The following pretrained models are required before training:

### 1. Chinese BERT (RoBERTa-wwm-ext-base)
```bash
mkdir -p ./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/
# Download from HuggingFace: hfl/chinese-roberta-wwm-ext-base
```

### 2. MAE Pretrained Weights
```bash
mkdir -p ./model_weights/
# Download MAE ViT-Base pretrained weights
# Place at: ./model_weights/mae_pretrain_vit_base.pth
```

### 3. Chinese CLIP Model
```bash
mkdir -p ./model_weights/clip_cn/
# cn_clip will auto-download, or specify the path manually
```

### 4. Word Vectors (Optional)
```bash
# Tencent AI Lab Chinese word vectors (for w2v mode)
# Place at: ./pretrained_model/w2v/
```

---

## Quick Start

### 0. Preprocess Images (required before first run, generates pkl files)

```bash
# Weibo -> data/
python preproc.py && python clipprep.py

# Weibo21 -> weibo21/
python w21prep.py && python w21clip.py

# FineFake -> FineFake/
python utils/extract.py
```

### Training

```bash
# Weibo (9 domains)
python main.py --dataset weibo --epoch 50 --batchsize 64 --lr 0.0001 --gpu 0

# Weibo21 (9 domains)
python main.py --dataset weibo21 --epoch 50 --batchsize 64 --lr 0.0001 --gpu 0

# FineFake (7 domains) - DDIN core model
python main.py --dataset finefake --model_name DDIN --epoch 50 --batchsize 64 --lr 0.0001 --gpu 0

# FineFake (7 domains) - GossipCop PLE-FEND model variant
python main.py --dataset finefake --model_name Gossip --epoch 50 --batchsize 64 --lr 0.0001 --gpu 0
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `DDIN` | Model: `DDIN` (core) or `Gossip` |
| `--dataset` | `weibo21` | Dataset: `weibo`, `weibo21`, `finefake` |
| `--epoch` | `50` | Number of training epochs |
| `--max_len` | `197` | Maximum text sequence length |
| `--batchsize` | `64` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--gpu` | `0` | GPU device ID |
| `--emb_type` | `bert` | Text embedding type: `bert` or `w2v` |
| `--early_stop` | `5` | Early stopping patience (epochs) |
| `--seed` | `3074` | Random seed for reproducibility |
| `--bert_emb_dim` | `768` | BERT embedding dimension (`--w2v_emb_dim` for w2v mode) |

---

## Dataset Format

### Weibo (`data/`) — 9 domains

Economy, Health, Military, Science, Politics, International, Education, Entertainment, Society

```
data/
├── train_origin.csv
├── val_origin.csv
├── test_origin.csv
├── nonrumor_images/
├── rumor_images/
├── train_loader.pkl
├── val_loader.pkl
├── test_loader.pkl
├── train_clip_loader.pkl
├── val_clip_loader.pkl
└── test_clip_loader.pkl
```

### Weibo21 (`weibo21/`) — 9 domains

Technology, Military, Education, Disaster, Politics, Healthcare, Finance, Entertainment, Society

```
weibo21/
├── train_datasets.xlsx
├── val_datasets.xlsx
├── test_datasets.xlsx
├── nonrumor_images/
├── rumor_images/
├── train_loader.pkl
├── val_loader.pkl
├── test_loader.pkl
├── train_clip_loader.pkl
├── val_clip_loader.pkl
└── test_clip_loader.pkl
```

### FineFake (`FineFake/`) — 7 domains

Politics, Entertainment, Business, Health, Society, Conflict, Uncategorized

```
FineFake/
├── FineFake.pkl                    # Main data (text + image paths + labels)
├── gossip_train.csv                # Training split (add 'category' column for domain labels)
├── gossip_test.csv                 # Test split
├── gossip_train/                   # Training images
├── f_train_loader.pkl              # MAE image features (from extract.py)
├── f_val_loader.pkl                # MAE image features (from extract.py)
├── f_test_loader.pkl               # MAE image features (from extract.py)
├── f_train_clip.pkl                # CLIP image features (from extract.py)
├── f_val_clip.pkl                  # CLIP image features (from extract.py)
└── f_test_clip.pkl                 # CLIP image features (from extract.py)
```

**Extract CLIP features from FineFake:**
```bash
python utils/extract.py
```
Encodes images with Chinese CLIP (ViT-B-16), saves as `f_train_loader.pkl` / `f_train_clip.pkl`.

### Dataset Category Mapping

| Dataset   | Domains | Categories |
|-----------|---------|------------|
| **Weibo** | 9 | Economy, Health, Military, Science, Politics, International, Education, Entertainment, Society |
| **Weibo21** | 9 | Technology, Military, Education, Disaster, Politics, Healthcare, Finance, Entertainment, Society |
| **FineFake** | 7 | Politics, Entertainment, Business, Health, Society, Conflict, Uncategorized |

> **Note:** DDIN dynamically sets `num_domains = len(category_dict)`, adapting to any number of categories.

---

## Training Techniques

| Technique | Description |
|-----------|-------------|
| **FGM Adversarial Training** | Applies perturbation to BERT embeddings to improve model robustness |
| **EMA (Exponential Moving Average)** | Smooths model parameters for better generalization |
| **Warmup + Cosine Annealing** | Linear warmup for the first 3 epochs, followed by cosine decay |
| **Layer-wise Learning Rate** | BERT layers use 0.1x base learning rate; other layers use full rate |
| **Multi-Task Auxiliary Loss** | Joint training with fusion, image, and text classifiers |
| **Adaptive Contrastive Loss** | Enhances cross-modal consistency learning |
| **Early Stopping** | Training halts when validation performance stops improving for N epochs |

---

## License

This project is intended for academic research purposes only. MIT License.
