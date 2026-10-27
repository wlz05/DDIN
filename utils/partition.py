# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

from collections import defaultdict
import hashlib
import html
import ntpath
from pathlib import Path
import posixpath
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_COLUMNS = ('text', 'content', 'post_text')
IMAGE_COLUMNS = ('image_path', 'image_id', 'image')
SOURCE_COLUMNS = ('post_id', 'news_id', 'article_id', 'source_id', 'event_id')

def paper_split(frame, seed=3407, image_root=None):
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError('Split seed must be an unsigned 32-bit integer')
    if 'label' not in frame or frame.empty:
        raise ValueError('Paper splits require nonempty binary annotations')
    labels = frame['label'].to_numpy()
    if labels.dtype.kind not in 'biuf' or not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError('Split labels must be finite binary values')
    # Follow the paper ratio without reallocating connected samples.
    train, remaining = train_test_split(frame, test_size=0.4, random_state=seed, stratify=labels)
    val, test = train_test_split(remaining, test_size=0.5, random_state=seed, stratify=remaining['label'])
    result = {'train': train.copy(), 'val': val.copy(), 'test': test.copy()}
    audit_frames(result, image_root)
    return result

def normalized_text(value):
    if pd.isna(value):
        return ''
    return ''.join(unicodedata.normalize('NFKC', html.unescape(str(value))).casefold().split())

def sample_keys(row, image_root=None, hashes=None, images=None):
    hashes = {} if hashes is None else hashes
    keys = set()
    for column in TEXT_COLUMNS:
        text = normalized_text(row.get(column, ''))
        if text:
            keys.add(('text', hashlib.sha256(text.encode('utf-8')).hexdigest()))
    for column in SOURCE_COLUMNS:
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            keys.add((column, str(value).strip()))
    for column in IMAGE_COLUMNS:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        for token in str(value).split('|'):
            token = token.strip().replace('\\', '/')
            if not token:
                continue
            if column == 'image_path':
                identity = posixpath.normpath(token).casefold()
            else:
                identity = token.split('?')[0].rsplit('/', 1)[-1].rsplit('.', 1)[0].casefold()
            keys.add(('image', identity))
            candidates = [Path(image_root) / token] if image_root is not None and column == 'image_path' else (images or {}).get(identity, [])
            for path in candidates:
                if path.is_file():
                    canonical = str(path.resolve())
                    if canonical not in hashes:
                        digest = hashlib.sha256()
                        with path.open('rb') as stream:
                            for chunk in iter(lambda: stream.read(1048576), b''):
                                digest.update(chunk)
                        hashes[canonical] = digest.hexdigest()
                    keys.add(('image_bytes', hashes[canonical]))
    return keys

def connected_rows(frame, image_root=None):
    if frame.empty or not any(column in frame.columns for column in TEXT_COLUMNS):
        raise ValueError('Dataset requires nonempty rows and a text column')
    parent, owners, hashes, images = list(range(len(frame))), {}, {}, defaultdict(list)
    if image_root is not None and any(column in frame for column in ('image_id', 'image')):
        for path in Path(image_root).rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'):
                images[path.stem.casefold()].append(path)
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    # Keep transitive text, image, and source links in one component.
    for index, (_, row) in enumerate(frame.iterrows()):
        for key in sample_keys(row, image_root, hashes, images):
            if key in owners:
                parent[find(index)] = find(owners[key])
            else:
                owners[key] = index
    groups = defaultdict(list)
    for index in range(len(frame)):
        groups[find(index)].append(index)
    return list(groups.values())

def audit_frames(frames, image_root=None):
    if set(frames) != {'train', 'val', 'test'} or any(frame is None or frame.empty for frame in frames.values()):
        raise ValueError('Train, validation, and test splits must all be present and nonempty')
    for name, frame in frames.items():
        if not any(column in frame.columns for column in TEXT_COLUMNS):
            raise ValueError('Missing text column in the ' + name + ' split')
    combined = pd.concat(list(frames.values()), ignore_index=True)
    membership = [name for name, frame in frames.items() for _ in range(len(frame))]
    for group in connected_rows(combined, image_root):
        if len({membership[index] for index in group}) > 1:
            raise ValueError('Text, image, or original-source leakage across dataset splits')
    for name, frame in frames.items():
        if 'label' not in frame:
            raise ValueError('Missing labels in the ' + name + ' split')
        labels = frame['label'].to_numpy()
        if labels.dtype.kind not in 'biuf' or not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
            raise ValueError('Dataset labels must be finite binary values')
    return {name: len(frame) for name, frame in frames.items()}

def audit_files(paths, image_root=None):
    frames = {}
    for name, value in paths.items():
        path = Path(value)
        if path.is_absolute() or ntpath.isabs(str(value)) or ntpath.splitdrive(str(value))[0]:
            raise ValueError('Dataset paths must be relative')
        frames[name] = pd.read_excel(path) if path.suffix.lower() in ('.xlsx', '.xls') else pd.read_csv(path)
    return audit_frames(frames, image_root)

def grouped_split(frame, seed=3407, ratios=(0.6, 0.2, 0.2), image_root=None):
    if 'label' not in frame:
        raise ValueError('Dataset metadata requires binary labels')
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError('Three positive split ratios must sum to one')
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError('Split seed must be an unsigned 32-bit integer')
    groups = connected_rows(frame, image_root)
    if len(groups) < 3:
        raise ValueError('At least three independent sample groups are required')
    labels = frame['label'].to_numpy()
    if labels.dtype.kind not in 'biuf' or not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError('Split labels must be finite binary values')
    targets = ratios[:, None] * np.bincount(labels.astype(int), minlength=2)
    counts, assignments = np.zeros((3, 2)), [[], [], []]
    np.random.default_rng(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    for order, group in enumerate(groups):
        histogram = np.bincount(labels[group].astype(int), minlength=2)
        empty = [index for index, assigned in enumerate(assignments) if not assigned]
        candidates = empty if len(groups) - order == len(empty) else range(3)
        costs = []
        for index in candidates:
            proposed = counts.copy()
            proposed[index] += histogram
            costs.append((float(np.sum((proposed - targets) ** 2 / np.maximum(targets, 1))), index))
        selected = min(costs)[1]
        counts[selected] += histogram
        assignments[selected].extend(group)
    result = {name: frame.iloc[indices].copy() for name, indices in zip(('train', 'val', 'test'), assignments)}
    audit_frames(result, image_root)
    return result

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
