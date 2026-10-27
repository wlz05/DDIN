# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

import argparse
import ntpath
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
if __package__:
    from .features import save_features
else:
    from features import save_features

ROOTS = {'weibo': 'data', 'weibo21': 'weibo21', 'finefake': 'FineFake'}

def encode_split(split, clip_preprocess, mae_transform, root='FineFake', dataset='finefake', view='both'):
    base = Path(root)
    if base.is_absolute() or ntpath.isabs(str(root)) or ntpath.splitdrive(str(root))[0]:
        raise ValueError('Image dataset root must be relative')
    if dataset not in ROOTS or split not in ('train', 'val', 'test') or view not in ('mae', 'clip', 'both'):
        raise ValueError('Unknown dataset, split, or preprocessing view')
    filename = split + ('_datasets.xlsx' if dataset == 'weibo21' else '_origin.csv' if dataset == 'weibo' else '.csv')
    metadata = base / filename
    frame = pd.read_excel(metadata) if dataset == 'weibo21' else pd.read_csv(metadata)
    if frame.empty:
        raise ValueError('Image preprocessing requires a nonempty split')
    images = {}
    if dataset != 'finefake':
        for directory in ('nonrumor_images', 'rumor_images'):
            for path in sorted((base / directory).glob('*')):
                if path.is_file():
                    images.setdefault(path.stem, path)
    streams = {name: [] for name in ('mae', 'clip') if view in (name, 'both')}
    missing = 0
    column = 'image_path' if dataset == 'finefake' else 'image_id' if dataset == 'weibo' else 'image'
    if column not in frame:
        raise ValueError('Image annotations require ' + column)
    # Keep exactly one transformed image per annotation row.
    for value in tqdm(frame[column], desc=split):
        photo = None
        if isinstance(value, str) and value.strip():
            if dataset == 'finefake':
                if ntpath.isabs(value) or ntpath.splitdrive(value)[0]:
                    raise ValueError('Image annotation paths must be relative')
                path = base / value
            else:
                path = next((images[key] for part in value.split('|') if (key := Path(part.strip()).stem) in images), None)
            if path is not None and path.is_file():
                try:
                    with Image.open(path) as source:
                        photo = source.convert('RGB')
                except (OSError, ValueError):
                    pass
        if photo is None:
            missing += 1
            photo = Image.new('RGB', (224, 224), (0, 0, 0))
        for name, stream in streams.items():
            stream.append((mae_transform if name == 'mae' else clip_preprocess)(photo))
    prefix = 'f_' if dataset == 'finefake' else ''
    for name, stream in streams.items():
        suffix = '_loader.pkl' if name == 'mae' else '_clip.pkl' if dataset == 'finefake' else '_clip_loader.pkl'
        save_features(base / (prefix + split + suffix), torch.stack(stream), metadata, view=name)
    print(split + ': ' + str(len(frame)) + ' rows; ' + str(missing) + ' placeholder images')

def main(dataset=None, view='both'):
    from torchvision import transforms
    from cn_clip.clip.utils import image_transform
    parser = argparse.ArgumentParser(description='Build per-split MAE and CLIP image caches')
    parser.add_argument('--dataset', choices=tuple(ROOTS), default=dataset or 'finefake')
    parser.add_argument('--root', default=None)
    parser.add_argument('--view', choices=('mae', 'clip', 'both'), default=view)
    args = parser.parse_args()
    mae = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for split in ('train', 'val', 'test'):
        encode_split(split, image_transform(224), mae, args.root or ROOTS[args.dataset], args.dataset, args.view)

if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
