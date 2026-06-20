# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DDIN** is a deep learning framework for **multimodal fake news detection**. It leverages a domain-aware disentanglement and interaction network to capture cross-modal inconsistencies between text and images, enabling robust identification of misinformation.

Paper Link: 

> Designed for fake news detection on multiple multimodal datasets including Weibo, Weibo-21, and FineFake.

---

## 🧠 Architecture

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

## 📁 Project Structure

```
DDIN/
├── model/
│   ├── net.py                 # DDIN core model + Trainer (dynamic num_domains) (dynamic num_domains)
│   ├── layers.py              # Base layers (MLP, Attention, FocalLoss, etc.)
│   ├── pivot.py               # Hypergraph convolution
│   ├── bert.py                # BERT modules
│   ├── domain.py              # Multi-domain PLE-FEND model variant
│   ├── weibo.py               # Weibo domain model variant
│   ├── w21.py                 # Weibo21 domain model variant
│   ├── gossip.py              # GossipCop/FineFake model (dynamic num_domains)
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
│   ├── clipld.py              # Weibo data loader (CSV -> data/)
│   ├── wloader.py             # Weibo CLIP image loader
│   ├── w21ld.py               # Weibo21 data loader (Excel -> Weibo21/)
│   ├── utils.py               # Metrics, Recorder, clipdata2gpu, data2gpu
│   ├── extract.py             # FineFake CLIP feature extraction -> FineFake/
│   ├── fsplit.py              # Fast data split
│   ├── fiximg.py              # Image fix utils
│   ├── datasets.py            # Dataset processing
│   ├── crop.py                # Image cropping
│   ├── lars.py                # LARS optimizer
│   ├── decay.py               # LR decay
│   ├── sched.py               # LR scheduling
│   ├── misc.py                # Miscellaneous utils
│   └── pos.py                 # Positional encoding
├── Weibo21/                 # Weibo21 dataset
│   ├── data.py                # Weibo21 data processing v1
│   ├── data2.py               # Weibo21 data processing v2
│   ├── probe.py               # Weibo21 experiment script
│   └── config.py              # Weibo21 configuration
├── main.py                    # Entry point (weibo / weibo21 / finefake) (argparse + config)
├── run.py                     # Training dispatch (3 datasets, DDIN + Gossip models) (weibo/weibo21 data loading)
├── mae.py                     # MAE ViT model (Masked Autoencoder)
├── dataset.py                 # FineFake/GossipCop dataset (category-aware, CSV auto-detect)
├── feature.py                 # t-SNE feature visualization (graceful fallback)
├── preproc.py                 # Weibo MAE image preprocessing -> data/
├── clipprep.py                # Weibo CLIP image preprocessing -> data/
├── w21prep.py                 # Weibo21 MAE image preprocessing -> Weibo21/
├── w21clip.py                 # Weibo21 CLIP image preprocessing -> Weibo21/
├── split.py                   # Reasoning column split utility
├── probe.py                   # Test probe
├── requirements.txt           # Dependencies (fixed: torch, timm, etc.)           # Dependencies (fixed format)
└── .gitignore
```

### Dataset Category Mapping

| Dataset   | Domains | Categories |
|-----------|---------|------------|
| **Weibo** | 9 | Economy, Health, Military, Science, Politics, International, Education, Entertainment, Society |
| **Weibo21** | 9 | Technology, Military, Education, Disaster, Politics, Healthcare, Finance, Entertainment, Society |
| **FineFake** | configurable | No built-in domain categories (binary classification per subset: gossip/politi) |

> **Note:** DDIN core model (`net.py`) dynamically adapts `num_domains` from `len(category_dict)`. GossipCop model (`gossip.py`) now also uses dynamic `num_domains` via `Trainer` injection.
## 🔧 Requirements

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
pip install -r requirements.txt           # Dependencies (fixed: torch, timm, etc.)
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

## 📥 Pretrained Models

The following pretrained models are required before training:

### 1. Chinese BERT (RoBERTa-wwm-ext-base)
```bash
mkdir -p ./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/
# Download from HuggingFace: hfl/chinese-roberta-wwm-ext-base
# https://huggingface.co/hfl/chinese-roberta-wwm-ext
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

## 🚀 Quick Start

> **Dataset folders:** `data/` (Weibo) · `Weibo21/` (Weibo21) · `FineFake/` (FineFake)

### 0. Preprocess Images (required before first run)

```bash
# Weibo -> data/
python preproc.py && python clipprep.py

# Weibo21 -> Weibo21/
python w21prep.py && python w21clip.py

# FineFake -> FineFake/
python utils/extract.py
```

### Training

```bash
# Weibo (9 domains) - DDIN core model
python main.py --dataset weibo --model_name DDIN --epoch 50 --batchsize 64 --lr 0.0001 --gpu 0

# Weibo21 (9 domains) - DDIN core model
python main.py --dataset weibo21 --model_name DDIN --epoch 50 --batchsize 64 --lr 0.0001 --gpu 0

# FineFake (6 domains) - DDIN core model
python main.py --dataset finefake --model_name DDIN --epoch 50 --batchsize 64 --lr 0.0001 --gpu 0

# FineFake (6 domains) - GossipCop PLE-FEND model variant
python main.py --dataset finefake --model_name Gossip --epoch 50 --batchsize 64 --lr 0.0001 --gpu 0
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `DDIN` | Model name (DDIN) |
| `--dataset` | `weibo21` | Dataset: `weibo`, `weibo21`, `finefake`, or `finefake` |
| `--epoch` | `50` | Number of training epochs |
| `--max_len` | `197` | Maximum text sequence length |
| `--batchsize` | `64` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--gpu` | `0` | GPU device ID |
| `--emb_type` | `bert` | Text embedding type: `bert` or `w2v` |
| `--early_stop` | `5` | Early stopping patience (epochs) |
| `--seed` | `42/1102/2026/3407` | Random seed for reproducibility |
| `--emb_dim` | `768` | Embedding dimension |

### Dataset Format

#### Weibo (`data/`)
```
data/
├── train_origin.csv
├── val_origin.csv
├── test_origin.csv
├── nonrumor_images/
└── rumor_images/
```

#### Weibo21 (`Weibo21/`)
```
Weibo21/
├── train_datasets.xlsx
├── val_datasets.xlsx
├── test_datasets.xlsx
├── nonrumor_images/
└── rumor_images/
```

#### FineFake (`FineFake/`)

FineFake is a multimodal fake news dataset with **6 semantic domains** (ref: [arXiv:2404.01336](https://arxiv.org/abs/2404.01336)): Politics, Entertainment, Business, Health, Society, Conflict.

```
FineFake/
├── FineFake.pkl                    # Main data (text + image paths + labels)
├── gossip_train.csv                # Training split (add 'category' column for domain labels)
├── gossip_test.csv                 # Test split
├── gossip_train/                   # Training images
├── f_train_loader.pkl              # MAE image features (from extract.py / fiximg.py)
└── f_train_clip.pkl                # CLIP image features (from extract.py)
```

**Extract CLIP features from FineFake:**
```bash
python utils/extract.py
```
Encodes images with Chinese CLIP (`ViT-B-16`), saves as `f_train_loader.pkl` / `f_train_clip.pkl`.

### Dataset Category Mapping

| Dataset   | Domains | Categories |
|-----------|---------|------------|
| **Weibo** | 9 | Economy, Health, Military, Science, Politics, International, Education, Entertainment, Society |
| **Weibo21** | 9 | Technology, Military, Education, Disaster, Politics, Healthcare, Finance, Entertainment, Society |
| **FineFake** | 6 | Politics, Entertainment, Business, Health, Society, Conflict |

> **Note:** DDIN core model dynamically sets `num_domains = len(category_dict)`. GossipCop model also uses dynamic `num_domains`. Both adapt to any number of categories.

---

## 🏗️ Training Techniques

| Technique | Description |
|-----------|-------------|
| **FGM Adversarial Training** | Applies perturbation to BERT embeddings to improve model robustness |
| **EMA (Exponential Moving Average)** | Smooths model parameters for better generalization |
| **Warmup + Cosine Annealing** | Linear warmup for the first 3 epochs, followed by cosine decay |
| **Layer-wise Learning Rate** | BERT layers use 0.1× base learning rate; other layers use full rate |
| **Multi-Task Auxiliary Loss** | Joint training with fusion, image, and text classifiers |
| **Adaptive Contrastive Loss** | Enhances cross-modal consistency learning |
| **Early Stopping** | Training halts when validation performance stops improving for N epochs |

---

## 📄 License

This project is intended for academic research purposes only. MIT License.
