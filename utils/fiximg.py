# -*-codeing = utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

import pandas as pd
import torch
import pickle
import os
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

root = './FineFake/'

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def process_split(split):
    pkl_path = root + f'{split}.pkl'
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"[ERROR] Data pickle not found: {pkl_path}")

    df = pd.read_pickle(pkl_path)
    images = []
    corrupted_count = 0
    missing_count = 0

    print(f"Restoring {split} set to 4D image tensors (3x224x224)...")

    for img_path in tqdm(df['image_path']):
        full_path = os.path.join(root, img_path)

        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            missing_count += 1
            if missing_count <= 5:
                print(f"[WARNING] Image file not found: {full_path}, using black placeholder")
            images.append(torch.zeros(3, 224, 224))
            continue

        try:
            im = Image.open(full_path).convert('RGB')
            im = data_transforms(im)
            images.append(im)
        except Exception as e:
            corrupted_count += 1
            if corrupted_count <= 5:
                print(f"[WARNING] Corrupted image: {full_path}, error: {e}")
            images.append(torch.zeros(3, 224, 224))

    if missing_count > 0:
        print(f"[INFO] {split}: {missing_count} missing images replaced with black placeholder")
    if corrupted_count > 0:
        print(f"[INFO] {split}: {corrupted_count} corrupted images replaced with black placeholder")

    final_tensor = torch.stack(images)
    output_path = root + f'f_{split}_loader.pkl'
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(final_tensor, f)
        print(f"[INFO] Saved {split} tensor ({final_tensor.shape}) to {output_path}")
    except Exception as e:
        raise IOError(f"[ERROR] Failed to save {output_path}: {e}")


if __name__ == '__main__':
    for split_name in ['train', 'val', 'test']:
        try:
            process_split(split_name)
        except Exception as e:
            print(f"[ERROR] Failed to process split '{split_name}': {e}")
    print("Image dimension fix complete! Real image features ready!")
