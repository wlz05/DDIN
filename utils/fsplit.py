# -*-codeing = utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

"""
FineFake official split generator.

The official FineFake repository (Accuser907/FineFake) ships a single
FineFake.pkl with NO predefined train/val/test files. The FineFake paper
(Section 6.2) splits the data 6:2:2 (train/val/test) with a fixed random
seed for the classification experiments. This script reproduces that
protocol:

  - stratified 60/20/20 split on the binary label (fixed seed, reproducible)
  - writes FineFake/train.csv, FineFake/val.csv, FineFake/test.csv
    (columns: text, image_path, label, category)
  - the three splits are disjoint by construction -> no leakage, and
    val is no longer identical to test (the old gossip_test.csv bug).

Run this BEFORE utils/extract.py.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

root = './FineFake/'
SEED = 3407  # fixed seed, as required by the FineFake protocol

print("[INFO] Loading FineFake data...")
if not os.path.exists(root + 'FineFake.pkl'):
    raise FileNotFoundError(f"[ERROR] FineFake metadata not found: {root + 'FineFake.pkl'}")

try:
    df = pd.read_pickle(root + 'FineFake.pkl')
except Exception as e:
    raise ValueError(f"[ERROR] Failed to load FineFake.pkl: {e}")

# Normalize schema -> text / image_path / label / category
required = ['text', 'label']
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"[ERROR] FineFake.pkl missing required columns: {missing}")

df = df.copy()
if 'image_path' not in df.columns:
    df['image_path'] = ""
df['category'] = df['topic'] if 'topic' in df.columns else 'Uncategorized'

# Drop rows with unusable text up front so CSVs stay aligned with features
before = len(df)
df['text'] = df['text'].fillna('').astype(str)
df = df[df['text'].str.strip().str.len() > 0].reset_index(drop=True)
if len(df) < before:
    print(f"[INFO] Dropped {before - len(df)} rows with empty text")

labels = df['label'].astype(int)
print(f"[INFO] Total samples: {len(df)}, fake (label=0): {(labels == 0).sum()}, real (label=1): {(labels == 1).sum()}")

# FineFake paper protocol: 6:2:2 train/val/test, fixed seed, stratified by label
train_df, tmp_df = train_test_split(
    df, test_size=0.4, random_state=SEED, stratify=labels)
tmp_labels = tmp_df['label'].astype(int)
val_df, test_df = train_test_split(
    tmp_df, test_size=0.5, random_state=SEED, stratify=tmp_labels)

out_cols = [c for c in ['text', 'image_path', 'label', 'category', 'platform', 'date'] if c in df.columns]
train_df[out_cols].to_csv(root + 'train.csv', index=False)
val_df[out_cols].to_csv(root + 'val.csv', index=False)
test_df[out_cols].to_csv(root + 'test.csv', index=False)

# Sanity check: splits must be pairwise disjoint
train_texts = set(train_df['text'])
val_texts = set(val_df['text'])
test_texts = set(test_df['text'])
assert not (train_texts & val_texts), "[FATAL] train/val text overlap detected"
assert not (train_texts & test_texts), "[FATAL] train/test text overlap detected"
assert not (val_texts & test_texts), "[FATAL] val/test text overlap detected"

total = len(train_df) + len(val_df) + len(test_df)
print(f"[INFO] Split (FineFake 6:2:2 protocol): train={len(train_df)} ({len(train_df)/total:.1%}), "
      f"val={len(val_df)} ({len(val_df)/total:.1%}), test={len(test_df)} ({len(test_df)/total:.1%})")
print("[INFO] Splits are pairwise disjoint, val != test. No leakage.")
print("[INFO] Saved train.csv / val.csv / test.csv to ./FineFake/")
print("[INFO] Next step: run utils/extract.py to build per-split image pkls.")

# Author:
# Corresponding Mail:
