# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

**DDIN** detects multimodal fake news through domain-aware disentanglement and interaction between text and image features. DDIN is accepted by [WISE 2026](https://conferences.sigappfr.org/wise2026/) (Research Paper).

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
│   ├── cache.py               # Relative model download caches
│   ├── partition.py           # Connected sample groups and partition verification
│   ├── extract.py             # Per-split MAE/CLIP preprocessing for all three datasets
│   ├── fsplit.py              # FineFake paper classification split (6:2:2)
│   ├── audit.py               # Partition verification for all three datasets
│   ├── fiximg.py              # Legacy image pkl fix utility (superseded by extract.py)
│   ├── datasets.py            # Dataset transforms (timm-based train/val augmentation)
│   ├── crop.py                # RandomResizedCrop + padding for MAE patching
│   ├── lars.py                # LARS optimizer with trust-coefficient scaling
│   ├── decay.py               # Weight-decay parameter grouping (exclude norm/bias)
│   ├── sched.py               # Cosine LR scheduler with linear warmup
│   ├── misc.py                # Distributed training utils, SmoothedValue, MetricLogger, NativeScaler
│   └── pos.py                 # Positional encoding (sinusoidal + learnable, 1D/2D)
├── verify/
│   └── boundary.py            # Metrics, training, image caches, and CLI regression tests
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

The experiment machine has **8 NVIDIA GeForce RTX 4090 GPUs**. The default experiment runs **four seeds: 42, 52, 3047, and 3407** sequentially. Use `--seeds 42 52 3047 3407` explicitly for the same complete grid, or `--seed 42` for a single run. Each command selects its GPU through `--gpu`.

```bash
python -B main.py --dataset weibo21 --gpu 0 --seed 42
python -B main.py --dataset weibo21 --gpu 0 --seed 52
python -B main.py --dataset weibo21 --gpu 0 --seed 3047
python -B main.py --dataset weibo21 --gpu 0 --seed 3407
```

Dataset partitions remain fixed across training seeds. FineFake uses partition seed 3407.

```bash
python -B main.py --dataset weibo21 --gpu 0 --seeds 42 52 3047 3407
```

Each seed trains independently, selects its checkpoint on validation, and evaluates that checkpoint on the complete test partition. Checkpoints and `result.json` use `./param_model/seed<seed>/<model_name>/`; `./param_model/summary.json` retains each seed's metrics in `byseed` and their equally weighted arithmetic means in `metrics[metric].mean`. `sample_std` uses the four seed results with denominator three. For every metric, the final mean is `(score42 + score52 + score3047 + score3407) / 4`. Predictions are not pooled or ensembled before metric calculation. A missing seed aborts the summary; a metric undefined for any seed has an undefined mean. Annotation hashes must stay identical across seeds. A custom `--save_param_dir` changes the common output root.

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

Model downloads use `./model_weights/huggingface` for Hugging Face assets, `./model_weights/clip_cn/` for Chinese-CLIP, and `./model_weights/checkpoints` for PyTorch checkpoints. Keep these directories with the project when moving an experiment to another machine. Model caches are excluded from Git.

---

## Quick Start

### Prepare Partitions and Image Caches

Place the released partition references alongside each dataset's prepared annotations as described in [Data Workflow](#data-workflow), then build the image caches:

```bash
python -B -m utils.extract --dataset weibo --root data
python -B -m utils.extract --dataset weibo21 --root weibo21
python -B -m utils.fsplit --root FineFake --seed 3407
python -B -m utils.extract --dataset finefake --root FineFake
python -B -m utils.audit
```

The original `preproc.py`, `clipprep.py`, `prepare.py`, and `encode.py` entry points call the same preprocessing functions.

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
| `--model_name` | `DDIN` | Model: `DDIN` (core) or `Gossip` (FineFake only) |
| `--root_path` | dataset default | Relative prepared dataset root |
| `--official_root` | dataset root | Relative directory containing released partition references |
| `--dataset` | `weibo21` | Dataset: `weibo`, `weibo21`, `finefake` |
| `--epoch` | `50` | Number of training epochs |
| `--max_len` | `128` | Maximum text sequence length |
| `--batchsize` | `64` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--gpu` | `0` | GPU device ID |
| `--emb_type` | `bert` | BERT tokenization |
| `--early_stop` | `5` | Early stopping patience (epochs) |
| `--seeds` | `42 52 3047 3407` | Independent training runs and equal-weight seed-metric aggregation |
| `--seed` | unset | Run one explicit seed instead of the default four-seed experiment |
| `--bert_emb_dim` | `768` | BERT embedding dimension |

The AdamW weight decay is **0.01** in the training configuration. Text preparation uses a maximum of **128 tokens**; CLIP follows its encoder context limit.

---

## Data Workflow

### 1. Data Partitions

| Dataset | Partition source | Prepared annotations |
| --- | --- | --- |
| [Weibo / EANN](https://github.com/yaqingwang/EANN-KDD18) | Released `train_id.pickle`, `validate_id.pickle`, and `test_id.pickle`; retain each post's assigned subset | `data/train_origin.csv`, `val_origin.csv`, `test_origin.csv` |
| [Weibo21 / MDFEND](https://github.com/kennqiang/MDFEND-Weibo21) | Released `train.pkl`, `val.pkl`, and `test.pkl`; retain the released subset when preparing multimodal Excel annotations | `weibo21/train_datasets.xlsx`, `val_datasets.xlsx`, `test_datasets.xlsx` |
| [FineFake](https://github.com/Accuser907/FineFake) | The paper's **6:2:2** classification ratio, generated once with this repository's split seed **3407** | `FineFake/train.csv`, `val.csv`, `test.csv` |

The [EANN preprocessing code](https://github.com/yaqingwang/EANN-KDD18/blob/master/src/process_data_weibo.py) selects Weibo membership by post ID. Keep `post_id` in the prepared CSVs. The [MDFEND loader](https://github.com/kennqiang/MDFEND-Weibo21/blob/main/utils/dataloader.py) reads its three released DataFrames and filters the undetermined category. Prepare Weibo21 annotations from the corresponding released subset, retaining content and labels while mapping the nine named domains to the English category names below. Put the reference pickles in the dataset root, or select their directory with `--official_root`. Training compares the prepared rows with these references before initializing a backbone.

The [FineFake paper, Section VI-B](https://arxiv.org/html/2404.01336#S6.SS2), specifies the classification ratio and a fixed random seed. `utils/fsplit.py` uses reproducible stratified 60/20/20 row partitions with seed 3407. Prepare the CSVs once and use the same files for all four training seeds. Weibo and Weibo21 retain **0 = real, 1 = fake**; FineFake retains **0 = fake, 1 = real**.

### 2. Image Caches

Create MAE and CLIP image tensors independently for `train`, `val`, and `test`. Each pickle contains the tensor, the annotation file's SHA-256, and its preprocessing view (`mae` or `clip`). Cache rows follow the annotation file's original order, including rows later removed by text filtering. The loader selects images using retained row indices. A missing or unreadable image receives a fixed black placeholder, preserving the row position. Rebuild both views whenever an annotation file changes.

Preprocessing uses fixed image transforms and normalization constants. BERT, MAE, and CLIP encode the inputs inside the model. The caches contain preprocessed images; preprocessing loads no learned statistics from the dataset. CLIP images use the [Chinese-CLIP transform](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/cn_clip/clip/utils.py).

### 3. Training Samples

The training loader draws only from `train`, with shuffling enabled. Validation and test loaders preserve their sample order and include the final partial batch. Valid text rows retain their original label, domain, and image alignment. The DDIN driver applies FGM during optimization and maintains EMA model weights. Training seeds change model initialization and training order while the three annotation files remain fixed.

### 4. Validation and Model Selection

After each epoch, evaluate the held-out `val` loader with gradients disabled. Select the checkpoint with the highest validation macro-F1 and stop after the configured patience. This selection rule follows the [MDFEND training implementation](https://github.com/kennqiang/MDFEND-Weibo21/blob/main/models/mdfend.py). The DDIN driver validates and saves EMA weights; the Gossip driver saves its validation-selected weights. The first valid epoch establishes the initial checkpoint.

### 5. Final Test

Reload the selected checkpoint, set evaluation mode, and process the complete `test` loader once. Report accuracy, macro precision, macro recall, macro-F1, ROC-AUC, and per-domain metrics from these predictions. Repeat the same fixed partition workflow for seeds **42, 52, 3047, and 3407**, then aggregate the four final test results.

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
├── train.csv                       # 60% train (fixed split seed 3407)
├── val.csv                         # 20% validation
├── test.csv                        # 20% test
├── f_train_loader.pkl              # MAE images [N,3,224,224] with split metadata hash
├── f_val_loader.pkl
├── f_test_loader.pkl
├── f_train_clip.pkl                # CLIP images [N,3,224,224] with split metadata hash
├── f_val_clip.pkl
└── f_test_clip.pkl
```

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
python -B -m unittest verify.boundary verify.protocol -v
```

The regression suite covers metrics, official partition references, fixed FineFake ratios, aligned image caches, preprocessing views, validation selection, checkpoint restoration, and the training-to-test sequence. Small tensor fixtures exercise the training loop without downloading pretrained backbones.

Custom dataset roots use relative paths through `main.py --root_path`, `utils.fsplit --root`, and `utils.extract --root`.

## Implementation References

- [UNet attention and timestep conditioning](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66)
- [BYOL crop transform](https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206)
- [BEiT parameter groups](https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58) and [layer assignment](https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33)
- [Supervised contrastive learning](https://arxiv.org/abs/2004.11362)
- [Focal loss](https://arxiv.org/abs/1708.02002)

## License

This project is intended for academic research purposes only. MIT License.
