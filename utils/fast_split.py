# DDIN: Domain-aware Disentanglement Interaction Network for Multimodal Fake News Detection

import pandas as pd
import torch
import pickle
import os
from sklearn.model_selection import train_test_split

root = '/root/autodl-tmp/FineFake_dataset/'


def safe_load_pickle(filepath, name="pickle"):
    """安全加载 pickle 文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] {name} not found: {filepath}")
    try:
        return pd.read_pickle(filepath) if filepath.endswith('.pkl') else pickle.load(open(filepath, 'rb'))
    except Exception as e:
        raise ValueError(f"[ERROR] Failed to load {name} from {filepath}: {e}")


def safe_save_pickle(data, filepath, name="pickle"):
    """安全保存 pickle 文件"""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    try:
        if hasattr(data, 'to_pickle'):
            data.to_pickle(filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
    except Exception as e:
        raise IOError(f"[ERROR] Failed to save {name} to {filepath}: {e}")


# 1. 加载总特征和总表
print("[INFO] Loading FineFake data...")
df = safe_load_pickle(root + 'FineFake.pkl', 'FineFake metadata')
full_features = safe_load_pickle(root + 'f_train_loader.pkl', 'full features')

if len(df) != len(full_features):
    raise ValueError(f"[ERROR] Row count mismatch: df has {len(df)} rows, features has {len(full_features)} rows")

# 2. 严谨切分 (8:1:1)
indices = list(range(len(df)))
train_idx, tmp_idx = train_test_split(indices, test_size=0.2, random_state=3407)
val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=3407)

print(f"[INFO] Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# 3. 保存分割表格
safe_save_pickle(df.iloc[train_idx], root + 'train.pkl', 'train metadata')
safe_save_pickle(df.iloc[val_idx], root + 'val.pkl', 'val metadata')
safe_save_pickle(df.iloc[test_idx], root + 'test.pkl', 'test metadata')


# 4. 分割特征矩阵
def save_feat(idx, prefix):
    """安全保存分割后的特征"""
    feat = full_features[idx]
    feat_path = root + f'{prefix}_loader.pkl'
    clip_path = root + f'{prefix}_clip.pkl'
    safe_save_pickle(feat, feat_path, f'{prefix} features')
    safe_save_pickle(feat, clip_path, f'{prefix} CLIP features')
    print(f"[INFO] Saved {len(idx)} samples to {feat_path}")


save_feat(train_idx, 'f_train')
save_feat(val_idx, 'f_val')
save_feat(test_idx, 'f_test')

print("✅ 一秒切分完成！文件全部对齐，绝无泄露！")
