import torch
import pandas as pd
import pickle
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

# --- 路径配置 ---
PKL_PATH = '/root/autodl-tmp/FineFake_dataset/FineFake.pkl'
IMG_DIR = '/root/autodl-tmp/FineFake_dataset/'
SAVE_IMAGE = '/root/autodl-tmp/FineFake_dataset/f_train_loader.pkl'
SAVE_CLIP = '/root/autodl-tmp/FineFake_dataset/f_train_clip.pkl'

# --- 加载模型 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载 CLIP 模型到 {device}...")
# 修正后的解包写法，不再报 tuple 错误
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()

# --- 读取数据 ---
df = pd.read_pickle(PKL_PATH)
features = []

print(f"开始提取 {len(df)} 张图片的特征...")

# --- 循环处理 ---
for i, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(IMG_DIR, row['image_path'])
    try:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image)
            feat = feat.cpu().numpy().flatten()
            features.append(feat)
    except Exception as e:
        features.append(np.zeros(512))

# --- 保存结果 ---
final_tensor = torch.tensor(np.array(features), dtype=torch.float32)
with open(SAVE_IMAGE, 'wb') as f:
    pickle.dump(final_tensor, f)
with open(SAVE_CLIP, 'wb') as f:
    pickle.dump(final_tensor, f)

print(f"\n✅ 特征提取完成！生成了 f_train_loader.pkl")