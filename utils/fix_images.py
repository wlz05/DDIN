# DDIN: Domain-aware Disentanglement Interaction Network for Multimodal Fake News Detection

import pandas as pd
import torch
import pickle
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

root = '/root/autodl-tmp/FineFake_dataset/'

# 严格按照 DDIN / MAE 要求的格式处理图片 (3通道, 224x224)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def process_split(split):
    df = pd.read_pickle(root + f'{split}.pkl')
    images = []
    print(f"正在将 {split} 集恢复为 4D 图像张量 (3x224x224)...")
    for img_path in tqdm(df['image_path']):
        full_path = os.path.join(root, img_path)
        try:
            im = Image.open(full_path).convert('RGB')
            images.append(data_transforms(im))
        except:
            # 损坏的图片用全零黑图占位，保证维度对齐不报错
            images.append(torch.zeros(3, 224, 224))

    # 堆叠成 [N, 3, 224, 224]
    final_tensor = torch.stack(images)
    with open(root + f'f_{split}_loader.pkl', 'wb') as f:
        pickle.dump(final_tensor, f)


process_split('train')
process_split('val')
process_split('test')
print("✅ 维度修复完毕！真正的图片特征已就绪！")
