# DDIN: Domain-Aware Disentanglement Interaction Network for Multimodal Fake News Detection

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

- **Multi-Granularity Inconsistency Mining**: Captures conflict signals between text and images at three granularities — Global-Global, Local-Local, and Global-Local.
- **Hierarchical Conflict Synergy**: A Transformer-based module that enables conflict features at different granularities to communicate and reinforce each other.
- **Domain-Adaptive Weighting**: Dynamically adjusts the importance of different inconsistency signals based on the news domain (9 categories: technology, military, education, etc.).
- **Multi-Scale Semantic Projection**: Employs multiple parallel projection channels to capture polysemous semantic correspondences between text and images.

---

## 📁 Project Structure

```
DDIN-code/
├── model/
│   ├── DDIN.py              # DDIN core model + Trainer
│   ├── layers.py             # Base layers (Embedding, MLP, etc.)
│   ├── pivot.py              # Hypergraph convolution and auxiliary modules
│   ├── bert.py               # BERT-related modules
│   ├── domain.py             # Domain processing
│   ├── domain_weibo.py       # Weibo dataset domain adaptation
│   ├── domain_weibo21.py     # Weibo21 dataset domain adaptation
│   ├── domain_gossipcop.py   # GossipCop dataset adaptation
│   ├── domain_raw.py         # Raw domain processing
│   ├── clip_domain.py        # CLIP domain module
│   └── test.py               # Test script
├── CNN_architectures/        # CNN architecture implementations
│   ├── pytorch_resnet.py     # ResNet
│   ├── pytorch_vgg_implementation.py  # VGG
│   ├── pytorch_efficientnet.py        # EfficientNet
│   ├── pytorch_inceptionet.py         # InceptionNet
│   ├── lenet5_pytorch.py     # LeNet-5
│   ├── unet.py               # U-Net
│   ├── nn.py                 # General network modules
│   └── fp16_util.py          # Mixed precision training utilities
├── utils/
│   ├── dataloader.py         # General data loader
│   ├── clip_dataloader.py    # CLIP data loader (Weibo)
│   ├── weibo_clip_dataloader.py      # Weibo CLIP loader
│   ├── weibo21_clip_dataloader.py    # Weibo21 CLIP loader
│   ├── utils.py              # Utility functions (metrics, recorder, etc.)
│   ├── extract-finefake.py   # FineFake data extraction
│   ├── fast_split.py         # Fast data splitting
│   └── fix_images.py         # Image fixing utilities
├── util/
│   ├── datasets.py           # Dataset processing
│   ├── crop.py               # Image cropping
│   ├── lars.py               # LARS optimizer
│   ├── lr_decay.py           # Learning rate decay
│   ├── lr_sched.py           # Learning rate scheduling
│   ├── misc.py               # Miscellaneous utilities
│   └── pos_embed.py          # Positional encoding
├── Weibo_21/                 # Weibo21 dataset utilities
│   ├── data.py               # Data processing
│   ├── data_2.py             # Data processing v2
│   ├── try.py                # Experiment script
│   └── variables.py          # Variable configuration
├── main.py                   # 🚀 Main entry point
├── run.py                    # Run controller (data loading + training dispatch)
├── models_mae.py             # MAE (Masked Autoencoder) model definition
├── FakeNet_dataset.py        # FakeNet dataset class
├── feature.py                # Feature extraction
├── data_pre.py               # Data preprocessing
├── clip_data_pre.py          # CLIP data preprocessing
├── fenge.py                  # Data splitting utility
├── weibo21_data_pre.py       # Weibo21 data preprocessing
├── weibo21_clip_data_pre.py  # Weibo21 CLIP preprocessing
├── requirements.txt          # Python dependencies
└── .gitignore                # Git ignore rules
```

---

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

### Training

```bash
# Train on Weibo, Weibo21, or FineFake
python main.py \
    --model_name DDIN \
    --dataset xxxxxx \
    --epoch 50 \
    --batchsize 64 \
    --lr 0.0001 \
    --gpu 0 \
    --emb_type bert \
    --early_stop 5
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `DDIN` | Model name (DDIN) |
| `--dataset` | `weibo21` | Dataset: `weibo`, `weibo21`, or `finefake` |
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

#### Weibo Dataset
```
./data/
├── train_origin.csv
├── val_origin.csv
└── test_origin.csv
```

#### Weibo21 Dataset
```
./Weibo_21/
├── train_datasets.xlsx
├── val_datasets.xlsx
└── test_datasets.xlsx
```

#### FineFake Dataset

[FineFake](https://github.com/Accuser907/FineFake) is a large-scale multimodal fake news dataset that covers diverse news topics with text-image pairs. The project supports both the `gossip` (GossipCop-style entertainment news) and `politi` (PolitiFact-style political news) subsets.

```
./FineFake_dataset/
├── FineFake.pkl                    # Main data file (text + image paths + labels)
├── gossip_train.csv                # GossipCop-style training split
├── gossip_test.csv                 # GossipCop-style test split
└── gossip_train/                   # Training images
```

**Extract CLIP features from FineFake:**
```bash
python utils/extract-finefake.py
```
This script loads images from the FineFake dataset, encodes them with Chinese CLIP (`ViT-B-16`), and saves the features as `f_train_loader.pkl` and `f_train_clip.pkl` for downstream training.

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
