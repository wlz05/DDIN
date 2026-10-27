# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

import argparse
from pathlib import Path
if __package__:
    from .partition import audit_files
else:
    from partition import audit_files

DATASETS = {
    'weibo': ('data', ('train_origin.csv', 'val_origin.csv', 'test_origin.csv')),
    'weibo21': ('weibo21', ('train_datasets.xlsx', 'val_datasets.xlsx', 'test_datasets.xlsx')),
    'finefake': ('FineFake', ('train.csv', 'val.csv', 'test.csv')),
}

def main():
    parser = argparse.ArgumentParser(description='Audit train, validation, and test leakage before training')
    parser.add_argument('--dataset', choices=(*DATASETS, 'all'), default='all')
    args = parser.parse_args()
    checked, failures = 0, 0
    for name, (root, filenames) in DATASETS.items():
        if args.dataset not in (name, 'all'):
            continue
        paths = {split: Path(root) / filename for split, filename in zip(('train', 'val', 'test'), filenames)}
        if args.dataset == 'all' and not any(path.exists() for path in paths.values()):
            print('[SKIP] ' + name + ': dataset files are absent')
            continue
        checked += 1
        try:
            counts = audit_files(paths, root)
            print('[OK] ' + name + ': disjoint text, images, and original sources; rows=' + str(counts))
        except (OSError, ValueError) as error:
            failures += 1
            print('[FAIL] ' + name + ': ' + str(error))
    if not checked:
        print('[FAIL] No dataset splits were available for auditing')
    return int(not checked or failures > 0)

if __name__ == '__main__':
    raise SystemExit(main())

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
