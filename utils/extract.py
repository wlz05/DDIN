# -*-codeing = utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

"""
FineFake per-split image preprocessing.

Builds the two image pkls each split needs, aligned row-by-row with the
split CSVs produced by utils/fsplit.py (FineFake 6:2:2 protocol):

  - f_{split}_loader.pkl : MAE-style transformed raw images [N, 3, 224, 224]
                           (same transforms as preproc.py for Weibo)
  - f_{split}_clip.pkl   : CLIP-preprocessed raw images      [N, 3, 224, 224]
                           (same as clipprep.py for Weibo)

The old version dumped 512-d CLIP *features* into BOTH files (and from the
FULL dataset into files named f_train_*): the MAE branch of the model
expects raw images and CLIP re-encodes its own input, so features in place
of raw images broke both branches and mixed up the two modalities.
Feature extraction happens inside the model now, same as Weibo/Weibo21.

Run utils/fsplit.py first.
"""

import torch
import pandas as pd
import pickle
import os
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from torchvision import transforms
from cn_clip.clip import load_from_name

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = './FineFake/'
SPLITS = ['train', 'val', 'test']

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Loading CLIP preprocessor on {device}...")
_, clip_preprocess = load_from_name("ViT-B-16", device=device, download_root='./')

# Identical to preproc.py / w21prep.py (Weibo / Weibo21 MAE branch input)
mae_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def encode_split(split):
    csv_path = os.path.join(ROOT, f'{split}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[ERROR] Split file not found: {csv_path}. Run utils/fsplit.py first.")

    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns:
        df['image_path'] = ""

    mae_images = []
    clip_images = []
    missing_count = 0
    corrupted_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{split}]"):
        rel = row['image_path']
        img_path = os.path.join(ROOT, str(rel)) if isinstance(rel, str) and rel.strip() else None

        img_pil = None
        if img_path is None or not os.path.isfile(img_path):
            missing_count += 1
        else:
            try:
                img_pil = Image.open(img_path).convert('RGB')
            except Exception:
                corrupted_count += 1

        if img_pil is None:
            mae_images.append(torch.zeros(3, 224, 224))
            clip_images.append(clip_preprocess(Image.new('RGB', (224, 224), (0, 0, 0))))
        else:
            mae_images.append(mae_transform(img_pil))
            clip_images.append(clip_preprocess(img_pil))

    mae_tensor = torch.stack(mae_images)
    clip_tensor = torch.stack(clip_images)
    assert mae_tensor.size(0) == len(df) and clip_tensor.size(0) == len(df)

    loader_path = os.path.join(ROOT, f'f_{split}_loader.pkl')
    clip_path = os.path.join(ROOT, f'f_{split}_clip.pkl')
    with open(loader_path, 'wb') as f:
        pickle.dump(mae_tensor, f)
    with open(clip_path, 'wb') as f:
        pickle.dump(clip_tensor, f)

    print(f"[{split}] {len(df)} rows -> {loader_path} {tuple(mae_tensor.shape)}, "
          f"{clip_path} {tuple(clip_tensor.shape)} "
          f"(missing: {missing_count}, corrupted: {corrupted_count})")


for split in SPLITS:
    encode_split(split)

print("[INFO] Done. Per-split MAE/CLIP image pkls are aligned with train/val/test.csv.")

# Author:
# Corresponding Mail:
