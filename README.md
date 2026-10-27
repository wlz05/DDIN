# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

**DDIN** is a deep learning framework for **multimodal fake news detection**. It leverages a domain-aware disentanglement and interaction network to capture cross-modal inconsistencies between text and images, enabling robust identification of misinformation. DDIN is accepted by [WISE 2026](https://conferences.sigappfr.org/wise2026/) (Research Paper).

Designed for fake news detection on multiple multimodal datasets including Weibo (9 domains), Weibo-21 (9 domains), and FineFake (7 domains).

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
- **Domain-Adaptive Weighting**: Dynamically adjusts the importance of different inconsistency signals based on the news domain (up to 9 categories per dataset, depending on the source; see [Dataset Category Mapping](#dataset-category-mapping)).
- **Multi-Scale Semantic Projection**: Employs multiple parallel projection channels to capture polysemous semantic correspondences between text and images.
- **Mixture of Experts (MoE)**: Multi-domain expert/gate networks route each domain's features through specialized and shared experts, with domain-specific gating for adaptive fusion.

---

## Project Structure

```
DDIN/
├── model/
│   ├── net.py                 # DDIN core model (BERT+MAE+CLIP) + Trainer (FGM, EMA, warmup-cosine)
│   ├── training.py            # Training state restoration, scheduling, and atomic checkpoints
│   ├── layers.py              # Base layers (MLP families, Attention, cnn_extractor, FocalLoss, SupConLoss, classifier)
│   ├── clip.py                # CLIP-based MoE domain model variant (MultiDomainPLEFENDModel + Trainer)
│   ├── raw.py                 # Raw DDIN model variant (domain-specific MoE experts, AdaIN)
│   ├── social.py              # Weibo21 domain model variant (same architecture as raw.py, different hyperparams)
│   ├── gossip.py              # FineFake/GossipCop PLE-FEND model (single-domain MoE with CLIP)
│   ├── domain.py              # Multi-domain PLE-FEND model variant (full-domain MoE)
│   ├── weibo.py               # Weibo domain model variant (manipulation classifier, gate/expert routing)
│   ├── bert.py                # BERT-only fake news detection baseline (BertFNModel + Trainer)
│   ├── pivot.py               # Session-based recommendation model (HypergraphConv + price/category Transformer)
│   └── test.py                # Quick smoke-test script (import-safe, no module-level execution)
├── cnn/
│   ├── resnet.py              # ResNet (18/34/50/101/152) with optional SRM (BayarConv) for forensic analysis
│   ├── vgg.py                 # VGG-16 / VGG-19
│   ├── efficient.py           # EfficientNet-style (MBConv + SqueezeExcitation + StochasticDepth)
│   ├── inception.py           # GoogLeNet / InceptionNet with optional SRM
│   ├── lenet.py               # LeNet-5
│   ├── unet.py                # U-Net / SuperResModel (diffusion backbone)
│   ├── nn.py                  # Shared network modules (SiLU, GroupNorm32, EMA, checkpoint)
│   └── precision.py           # Mixed-precision conversion utils (fp16↔fp32, master param sync)
├── utils/
│   ├── loader.py              # Generic data loader (Weibo CSV → TensorDataset)
│   ├── clipld.py              # Weibo CLIP-MAE data loader (text + image pkls)
│   ├── social.py              # Weibo21 CLIP-MAE data loader (xlsx → TensorDataset)
│   ├── fld.py                 # FineFake CLIP-MAE data loader (CSV → TensorDataset)
│   ├── utils.py               # Metrics (AUC, precision/recall/F1), Recorder, clipdata2gpu, data2gpu
│   ├── features.py            # Binary labels, bound image caches, and CLIP token fallback
│   ├── partition.py           # Connected sample groups and mandatory split leakage checks
│   ├── extract.py             # FineFake per-split MAE/CLIP image preprocessing (generates pkl files)
│   ├── fsplit.py              # FineFake grouped split (60/20/20 targets, seed 3407)
│   ├── audit.py               # Split/leakage audit for all three datasets
│   ├── fiximg.py              # Legacy image pkl fix utility (superseded by extract.py)
│   ├── datasets.py            # Dataset transforms (timm-based train/val augmentation)
│   ├── crop.py                # RandomResizedCrop + padding for MAE patching
│   ├── lars.py                # LARS optimizer with trust-coefficient scaling
│   ├── decay.py               # Weight-decay parameter grouping (exclude norm/bias)
│   ├── sched.py               # Cosine LR scheduler with linear warmup
│   ├── misc.py                # Distributed training utils, SmoothedValue, MetricLogger, NativeScaler
│   └── pos.py                 # Positional encoding (sinusoidal + learnable, 1D/2D)
├── verify/
│   └── boundary.py            # Metrics, training failures, cached images, and CLI regression tests
├── data/                      # Weibo dataset (CSV + generated pkls)
│   ├── train_origin.csv
│   ├── val_origin.csv
│   ├── test_origin.csv
│   ├── nonrumor_images/
│   └── rumor_images/
├── weibo21/                   # Weibo21 dataset (Excel + images + generated pkls)
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
├── social/                    # Weibo21 data preparation utilities
│   ├── data.py                # Image download + text preprocessing
│   ├── download.py            # Batch image download script
│   ├── probe.py               # Data statistics probe
│   └── config.py              # Text cleaning (emoji removal, fullwidth→halfwidth)
├── main.py                    # Entry point (argparse + config build)
├── run.py                     # Training dispatch (3 datasets, DDIN + Gossip model selection, loader routing)
├── mae.py                     # MAE ViT model (base/large/huge variants, encoder/decoder)
├── dataset.py                 # FineFake/GossipCop dataset class (pos_weight logging)
├── feature.py                 # t-SNE feature visualization (requires trained model weights)
├── preproc.py                 # Weibo MAE image preprocessing → data/
├── clipprep.py                # Weibo CLIP image preprocessing → data/
├── prepare.py                 # Weibo21 MAE image preprocessing → weibo21/
├── encode.py                  # Weibo21 CLIP image preprocessing → weibo21/
├── split.py                   # Reasoning column split utility (parses "1)\2)\3)" annotations)
├── probe.py                   # Model probe for shape verification
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
| numpy | 1.23.2 |
| transformers | latest |
| cn_clip | latest |
| openai/CLIP | latest (git) |
| timm | latest |
| positional_encodings | latest |
| open_clip_torch | latest |
| scikit-learn | latest |
| pandas | latest |
| openpyxl | latest |
| matplotlib | latest |
| seaborn | latest |
| tqdm | latest |
| ftfy | latest |
| regex | latest |

### Installation

```bash
pip install -r requirements.txt
```

Key dependencies:

- `torch==2.1.0` — Deep learning framework
- `transformers` — Pre-trained models (BERT, RoBERTa)
- `cn_clip` — Chinese CLIP model (Weibo/Weibo21 global features)
- `openai/CLIP` (`git+https://github.com/openai/CLIP.git`) — OpenAI CLIP model (FineFake)
- `open_clip_torch` — OpenCLIP model loading (Gossip model variant)
- `timm` — Vision Transformer and model components
- `positional_encodings` — Positional encoding utilities
- `scikit-learn` — Metrics and t-SNE visualization
- `matplotlib`, `seaborn` — Visualization (feature.py)
- `pandas`, `openpyxl` — Data processing (CSV + Excel)

---

## Reproducibility

Experiments use one NVIDIA GeForce RTX 4090 and seeds **42, 52, 3047, and 3407**. The default training seed is 42.

```bash
python -B main.py --dataset weibo21 --gpu 0 --seed 42
python -B main.py --dataset weibo21 --gpu 0 --seed 52
python -B main.py --dataset weibo21 --gpu 0 --seed 3047
python -B main.py --dataset weibo21 --gpu 0 --seed 3407
```

Dataset partitions remain fixed across training seeds. FineFake uses partition seed 3407.

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
# cn_clip will auto-download on first use, or specify the path manually
```

### 4. Word Vectors (Optional, w2v mode &#8212; dataset-dependent)

```bash
# Tencent AI Lab Chinese word vectors
# Place at: ./pretrained_model/w2v/
```

---

## Quick Start

### 0. Preprocess Images (required before first run, generates pkl files)

```bash
# Weibo -> data/
python preproc.py && python clipprep.py

# Weibo21 -> weibo21/
python prepare.py && python encode.py

# FineFake -> FineFake/  (1: grouped split, 2: bound image caches)
python utils/fsplit.py && python utils/extract.py
```

### 0.1 Verify splits / check for leakage (recommended)

```bash
python utils/audit.py
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

> **Note on emb_type:** The `--emb_type` flag accepts `bert` (default) and `w2v` values.
> Currently the `w2v` path requires a custom data loader (not included); use `bert` mode
> for standard training.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `DDIN` | Model: `DDIN` (core) or `Gossip` (FineFake only) |
| `--dataset` | `weibo21` | Dataset: `weibo`, `weibo21`, `finefake` |
| `--epoch` | `50` | Number of training epochs |
| `--max_len` | `197` | Maximum text sequence length |
| `--batchsize` | `64` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--gpu` | `0` | GPU device ID |
| `--emb_type` | `bert` | Text embedding type: `bert` (recommended) or `w2v` |
| `--early_stop` | `5` | Early stopping patience (epochs) |
| `--seed` | `42` | Training seed; experimental seeds: 42, 52, 3047, 3407 |
| `--bert_emb_dim` | `768` | BERT embedding dimension (`--w2v_emb_dim` for w2v mode) |

---

## Dataset Format & Split Provenance

DDIN uses Weibo, Weibo21, and FineFake for multimodal fake news detection. The loaders read prepared CSV or Excel annotations and the corresponding image caches.

| Dataset | Data and split | Files |
|---------|--------------|-------|
| [Weibo](https://github.com/yaqingwang/EANN-KDD18) | Weibo benchmark with prepared train/validation/test CSV annotations | `data/train_origin.csv`, `val_origin.csv`, `test_origin.csv` |
| [Weibo21](https://github.com/kennqiang/MDFEND-Weibo21) | MDFEND-Weibo21 release with prepared train/validation/test Excel annotations | `weibo21/train_datasets.xlsx`, `val_datasets.xlsx`, `test_datasets.xlsx` |
| [FineFake](https://github.com/Accuser907/FineFake) | Connected text/image/source groups with **60/20/20 target ratios**, label balancing, and partition seed 3407 | `FineFake/train.csv`, `val.csv`, `test.csv` |

Validation controls early stopping; the test split is used for final evaluation. Training audits normalized text, image identifiers, available image-file hashes, and original-source IDs before loading model backbones. Cross-split overlap stops the run. Official Weibo and Weibo21 files are audited without repartitioning. FineFake groups stay intact, so achieved ratios follow group sizes rather than an exact row-level split. Run `python -B utils/audit.py --dataset finefake` to inspect the selected dataset.

Weibo and Weibo21 use **1 = fake, 0 = real**. FineFake retains the released binary labels: **0 = fake, 1 = real**.

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
├── train.csv                       # 60% target (connected groups, seed 3407)
├── val.csv                         # 20% target (disjoint from train and test)
├── test.csv                        # 20% target (disjoint from train and val)
├── f_train_loader.pkl              # MAE images [N,3,224,224] with split metadata hash
├── f_val_loader.pkl
├── f_test_loader.pkl
├── f_train_clip.pkl                # CLIP images [N,3,224,224] with split metadata hash
├── f_val_clip.pkl
└── f_test_clip.pkl
```

**Build FineFake splits + image pkls:**
```bash
python utils/fsplit.py     # FineFake.pkl -> grouped train/val/test.csv (60/20/20 targets, seed 3407)
python utils/extract.py    # csvs -> aligned per-split MAE/CLIP image pkls
```
Each FineFake pickle stores raw preprocessed images and the SHA-256 of its split CSV. The loader rejects a cache created for different metadata. Rebuild both image caches after changing or regenerating the split files. BERT/MAE/CLIP encoding happens inside the model at training time. Image preprocessing uses the [Chinese-CLIP transform](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/cn_clip/clip/utils.py).

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
| **Layer-wise Learning Rate** | BERT layers use 0.1× base learning rate; other layers use full rate |
| **Multi-Task Auxiliary Loss** | Joint training with fusion, image, and text auxiliary classifiers |
| **Domain-Adaptive Contrastive Loss** | Enhances cross-modal consistency learning with domain-aware weighting |
| **Mixture of Experts (MoE)** | Per-domain specific + shared experts with learned gating networks |
| **Early Stopping** | Training halts when validation performance stops improving for N epochs |

---

## Tests

```bash
python -B -m unittest verify.boundary -v
```

The regression suite covers metric edge cases, missing text and image-cache alignment, CLIP token fallback, single-epoch training, FGM/EMA restoration, checkpoint write failures, and command-line validation. Small tensor fixtures exercise the training loop and model dimensions without downloading pretrained backbones. Partition tests cover transitive grouping, normalized duplicate text, copied image content, stale caches, and audit failure before model loading.

Custom dataset roots use relative paths through `main.py --root_path`, `utils/fsplit.py --root`, and `utils/extract.py --root`.

Image caches follow the original metadata row order. When text filtering removes rows, the loaders select the corresponding cached images. CLIP fallback text uses the tokenizer's context length. Checkpoints are replaced after a complete write, and the first valid validation epoch is saved even when F1 is zero.

## License

This project is intended for academic research purposes only. MIT License.
