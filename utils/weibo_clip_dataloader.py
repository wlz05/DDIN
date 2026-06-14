# DDIN: Domain-aware Disentanglement Interaction Network for Multimodal Fake News Detection

import torch
import pandas as pd
import pickle
import os
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

# 允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 路径配置 ---
PKL_PATH = '/root/autodl-tmp/FineFake_dataset/FineFake.pkl'
IMG_DIR = '/root/autodl-tmp/FineFake_dataset/'
SAVE_IMAGE = '/root/autodl-tmp/FineFake_dataset/f_train_loader.pkl'
SAVE_CLIP = '/root/autodl-tmp/FineFake_dataset/f_train_clip.pkl'

# --- 加载模型 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载 CLIP 模型到 {device}...")
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()

# --- 读取数据 ---
if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(f"[ERROR] FineFake pickle not found: {PKL_PATH}")

try:
    df = pd.read_pickle(PKL_PATH)
except Exception as e:
    raise ValueError(f"[ERROR] Failed to read FineFake pickle {PKL_PATH}: {e}")

features = []
corrupted_count = 0
missing_count = 0

print(f"开始提取 {len(df)} 张图片的特征...")

# --- 循环处理 ---
for i, row in tqdm(df.iterrows(), total=len(df)):
    # 检查 image_path 是否有效
    if 'image_path' not in row or pd.isna(row['image_path']):
        missing_count += 1
        features.append(np.zeros(512))
        continue

    img_path = os.path.join(IMG_DIR, row['image_path'])

    # 检查文件是否存在
    if not os.path.exists(img_path) or not os.path.isfile(img_path):
        missing_count += 1
        if missing_count <= 5:
            print(f"[WARNING] Image file not found: {img_path}")
        features.append(np.zeros(512))
        continue

    try:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image)
            feat = feat.cpu().numpy().flatten()
            features.append(feat)
    except Exception as e:
        corrupted_count += 1
        if corrupted_count <= 5:
            print(f"[WARNING] Failed to process image: {img_path}, error: {e}")
        features.append(np.zeros(512))

# --- 保存结果 ---
if missing_count > 0:
    print(f"[INFO] {missing_count} missing images replaced with zero vectors")
if corrupted_count > 0:
    print(f"[INFO] {corrupted_count} corrupted images replaced with zero vectors")

final_tensor = torch.tensor(np.array(features), dtype=torch.float32)
try:
    os.makedirs(os.path.dirname(SAVE_IMAGE), exist_ok=True)
    with open(SAVE_IMAGE, 'wb') as f:
        pickle.dump(final_tensor, f)
    with open(SAVE_CLIP, 'wb') as f:
        pickle.dump(final_tensor, f)
except Exception as e:
    raise IOError(f"[ERROR] Failed to save feature files: {e}")

print(f"\n✅ 特征提取完成！保存了 f_train_loader.pkl ({final_tensor.shape[0]} 张图片特征)")
