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
│   ├── w21ld.py               # Weibo21 data loader
│   ├── fld.py                 # FineFake data loader
│   ├── utils.py               # Metrics, Recorder, clipdata2gpu, data2gpu
│   ├── extract.py             # FineFake per-split MAE/CLIP image preprocessing
│   ├── fsplit.py              # FineFake official-protocol split (6:2:2, fixed seed)
│   ├── check_splits.py        # Split/leakage audit for all three datasets
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
│   ├── train.csv / val.csv / test.csv
│   ├── f_{train,val,test}_loader.pkl
│   └── f_{train,val,test}_clip.pkl
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

# FineFake -> FineFake/  (1: official-protocol split, 2: per-split image pkls)
python utils/fsplit.py && python utils/extract.py
```

### 0.1 Verify splits / check for leakage (recommended)

```bash
python utils/check_splits.py
```

Prints per-split counts, label/domain distributions, and cross-split
overlap checks (identical texts, shared post IDs, shared images) for all
three datasets. Exits non-zero if any leakage is detected.

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

## Dataset Format & Split Provenance

Each dataset uses the split defined by its source paper:

| Dataset | Split source | Files |
|---------|--------------|-------|
| **Weibo** | Official benchmark split of Jin et al. (MM 2017), as used by EANN/CAFE/MCAN and follow-ups | `data/train_origin.csv`, `val_origin.csv`, `test_origin.csv` |
| **Weibo21** | Official split released with MDFEND-Weibo21 (Nan et al., CIKM 2021) | `weibo21/train_datasets.xlsx`, `val_datasets.xlsx`, `test_datasets.xlsx` |
| **FineFake** | No official split files exist; the paper (Zhou et al., Inf. Fusion) prescribes a **6:2:2 train/val/test** split with a fixed seed — reproduced by `utils/fsplit.py` (seed 3407, stratified by label) | `FineFake/train.csv`, `val.csv`, `test.csv` |

> **Leakage note:** an earlier version trained FineFake on `gossip_train.csv`
> and used `gossip_test.csv` as **both** validation and test set, leaking the
> test set into early stopping/model selection. This has been replaced by the
> disjoint 6:2:2 protocol above; `run.py` now refuses to start if the val and
> test paths are identical. Run `python utils/check_splits.py` to audit.

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
├── FineFake.pkl                    # Official full data (text + image_path + label + topic, no splits)
├── train.csv                       # 60% (generated by utils/fsplit.py, 6:2:2 protocol)
├── val.csv                         # 20% (disjoint from train and test)
├── test.csv                        # 20% (disjoint from train and test)
├── f_train_loader.pkl              # MAE-branch images [N,3,224,224] (from utils/extract.py)
├── f_val_loader.pkl
├── f_test_loader.pkl
├── f_train_clip.pkl                # CLIP-preprocessed images [N,3,224,224]
├── f_val_clip.pkl
└── f_test_clip.pkl
```

**Build FineFake splits + image pkls:**
```bash
python utils/fsplit.py     # FineFake.pkl -> train/val/test.csv (6:2:2, seed 3407, stratified)
python utils/extract.py    # csvs -> aligned per-split MAE/CLIP image pkls
```
Both pkls hold raw preprocessed images (as for Weibo/Weibo21); BERT/MAE/CLIP
encoding happens inside the model. Labels follow the official convention:
`0 = fake`, `1 = real`; `topic` is mapped to the `category` column.

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
