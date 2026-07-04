# -*-codeing = utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

import pandas as pd
import torch
import pickle
import os
from sklearn.model_selection import train_test_split

root = './FineFake/'

def safe_load_pickle(filepath, name="pickle"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] {name} not found: {filepath}")
    try:
        return pd.read_pickle(filepath) if filepath.endswith('.pkl') else pickle.load(open(filepath, 'rb'))
    except Exception as e:
        raise ValueError(f"[ERROR] Failed to load {name} from {filepath}: {e}")

def safe_save_pickle(data, filepath, name="pickle"):
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    try:
        if hasattr(data, 'to_pickle'):
            data.to_pickle(filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
    except Exception as e:
        raise IOError(f"[ERROR] Failed to save {name} to {filepath}: {e}")

print("[INFO] Loading FineFake data...")
df = safe_load_pickle(root + 'FineFake.pkl', 'FineFake metadata')
full_features = safe_load_pickle(root + 'f_train_loader.pkl', 'full features')

if len(df) != len(full_features):
    raise ValueError(f"[ERROR] Row count mismatch: df has {len(df)} rows, features has {len(full_features)} rows")

indices = list(range(len(df)))
train_idx, tmp_idx = train_test_split(indices, test_size=0.2, random_state=3407)
val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=3407)

print(f"[INFO] Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

safe_save_pickle(df.iloc[train_idx], root + 'train.pkl', 'train metadata')
safe_save_pickle(df.iloc[val_idx], root + 'val.pkl', 'val metadata')
safe_save_pickle(df.iloc[test_idx], root + 'test.pkl', 'test metadata')

def save_feat(idx, prefix):
    feat = full_features[idx]
    feat_path = root + f'{prefix}_loader.pkl'
    clip_path = root + f'{prefix}_clip.pkl'
    safe_save_pickle(feat, feat_path, f'{prefix} features')
    safe_save_pickle(feat, clip_path, f'{prefix} CLIP features')
    print(f"[INFO] Saved {len(idx)} samples to {feat_path}")

save_feat(train_idx, 'f_train')
save_feat(val_idx, 'f_val')
save_feat(test_idx, 'f_test')

print("Split complete! All files aligned, no leakage!")

# Author: 
# Corresponding Mail: 
