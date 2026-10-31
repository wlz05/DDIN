# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

from collections import Counter
import pickle
import ntpath
from pathlib import Path
import pandas as pd
from utils.features import metadata_hash
from utils.partition import normalized_text

def verify_files(dataset, paths, source_root):
    root = Path(source_root)
    if dataset not in ('weibo', 'weibo21'):
        return {}
    if root.is_absolute() or str(source_root).startswith(("/", "\\")) or ntpath.isabs(str(source_root)) or ntpath.splitdrive(str(source_root))[0]:
        raise ValueError('Official reference root must be relative')
    sources = {'train': 'train_id.pickle', 'val': 'validate_id.pickle', 'test': 'test_id.pickle'} if dataset == 'weibo' else {name: name + '.pkl' for name in paths}
    signatures = {}
    for name, value in paths.items():
        source = root / sources[name]
        if not source.is_file():
            raise FileNotFoundError('Provide the released reference partition: ' + str(source))
        frame = pd.read_excel(value) if Path(value).suffix.lower() in ('.xlsx', '.xls') else pd.read_csv(value, dtype={'post_id': 'string'})
        with source.open('rb') as stream:
            reference = pickle.load(stream)
        if dataset == 'weibo':
            if 'post_id' not in frame or frame['post_id'].isna().any():
                raise ValueError('Weibo annotations must preserve the released post IDs')
            allowed = {item.decode('ascii') if isinstance(item, bytes) else str(item) for item in reference}
            actual = frame['post_id'].astype(str).tolist()
            if len(set(actual)) != len(actual) or not set(actual).issubset(allowed):
                raise ValueError('Prepared rows differ from the released ' + name + ' IDs')
        else:
            if not isinstance(reference, pd.DataFrame) or not {'content', 'label'}.issubset(reference):
                raise ValueError('MDFEND reference partitions must be annotation DataFrames')
            if not {'content', 'label'}.issubset(frame):
                raise ValueError('Prepared Weibo21 rows require content and label')
            allowed = Counter((normalized_text(text), int(label)) for text, label in zip(reference['content'], reference['label']))
            actual = Counter((normalized_text(text), int(label)) for text, label in zip(frame['content'], frame['label']))
            if actual - allowed:
                raise ValueError('Prepared rows differ from the released ' + name + ' annotations')
        signatures[name] = metadata_hash(source)
    return signatures

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
