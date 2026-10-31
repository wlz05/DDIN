# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

from pathlib import Path
import argparse
import ntpath
import os
import tempfile
import pandas as pd
if __package__:
    from .partition import paper_split, SOURCE_COLUMNS
else:
    from partition import paper_split, SOURCE_COLUMNS

SEED = 3407

def prepare(root='FineFake', seed=SEED):
    base = Path(root)
    if base.is_absolute() or str(root).startswith(("/", "\\")) or ntpath.isabs(str(root)) or ntpath.splitdrive(str(root))[0]:
        raise ValueError('FineFake root must be relative')
    frame = pd.read_pickle(base / 'FineFake.pkl')
    if not isinstance(frame, pd.DataFrame) or not {'text', 'label'}.issubset(frame.columns):
        raise ValueError('FineFake metadata requires text and label columns')
    frame = frame.copy()
    frame['text'] = frame['text'].fillna('').astype(str)
    frame = frame[frame['text'].str.strip().str.len() > 0].reset_index(drop=True)
    if 'image_path' not in frame:
        frame['image_path'] = ''
    if 'category' not in frame:
        frame['category'] = frame['topic'] if 'topic' in frame else 'Uncategorized'
    splits = paper_split(frame, seed=seed, image_root=base)
    columns = [name for name in ('text', 'image_path', 'label', 'category', 'platform', 'date', *SOURCE_COLUMNS) if name in frame]
    # Validate every split before staging any output file.
    with tempfile.TemporaryDirectory(prefix='.split-', dir=base) as directory:
        staged = Path(directory)
        for name, subset in splits.items():
            subset[columns].to_csv(staged / (name + '.csv'), index=False)
        for name in splits:
            os.replace(staged / (name + '.csv'), base / (name + '.csv'))
    return splits

def main():
    parser = argparse.ArgumentParser(description='Build FineFake paper-protocol splits')
    parser.add_argument('--root', default='FineFake')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    splits = prepare(args.root, args.seed)
    total = sum(len(frame) for frame in splits.values())
    print('FineFake classification split: 0.6/0.2/0.2; seed: ' + str(args.seed))
    for name, frame in splits.items():
        print(name + ': ' + str(len(frame)) + ' rows (' + format(len(frame) / total, '.1%') + ')')
    print('Rebuild per-split image caches with python utils/extract.py before training.')

if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
