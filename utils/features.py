# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

import pickle
import hashlib
import tempfile
import torch
import numpy as np
import ntpath
import os
from pathlib import Path

def binary_labels(values):
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.dtype.kind not in 'biuf' or not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError('Labels must be finite binary values')
    return torch.tensor(labels.astype(np.int64), dtype=torch.long)

def metadata_hash(path):
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip() or Path(path).is_absolute() or str(path).startswith(("/", "\\")) or ntpath.isabs(str(path)) or ntpath.splitdrive(str(path))[0]:
        raise ValueError('Metadata path must be a nonempty relative path')
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save_features(path, values, metadata, view=None):
    target = Path(path)
    if target.is_absolute() or str(path).startswith(("/", "\\")) or ntpath.isabs(str(path)) or ntpath.splitdrive(str(path))[0] or target.resolve() == Path(metadata).resolve():
        raise ValueError('Feature output must be relative and differ from its metadata')
    if view not in (None, 'mae', 'clip'):
        raise ValueError('Unknown image cache view')
    payload = {'version': 1, 'metadata_hash': metadata_hash(metadata), 'view': view, 'images': values}
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.features-', dir=target.parent)
    try:
        with os.fdopen(handle, 'wb') as stream:
            pickle.dump(payload, stream)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def clip_tokens(texts, tokenizer):
    texts = list(texts)
    if not texts:
        raise ValueError('CLIP tokenization requires nonempty text input')
    try:
        tokens = tokenizer(texts)
    except Exception:
        rows = []
        for text in texts:
            try:
                rows.append(tokenizer([str(text)])[0])
            except Exception:
                try:
                    # Keep the tokenizer's actual context length for fallback text.
                    rows.append(tokenizer([' '])[0])
                except Exception as error:
                    raise ValueError('CLIP tokenization failed for both input and fallback text') from error
        try:
            tokens = torch.stack(rows)
        except (TypeError, RuntimeError) as error:
            raise ValueError('CLIP tokenizer returned inconsistent token rows') from error
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or len(tokens) != len(texts) or tokens.shape[1] < 2 or tokens.dtype != torch.long or torch.any(tokens < 0):
        raise ValueError('CLIP tokenizer must return a nonnegative integer token matrix')
    return tokens

def read_features(path, indices, total_rows, metadata=None, require_binding=False, view=None):
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip() or Path(path).is_absolute() or str(path).startswith(("/", "\\")) or ntpath.isabs(str(path)) or ntpath.splitdrive(str(path))[0]:
        raise ValueError('Image cache path must be a nonempty relative path')
    rows = np.asarray(indices)
    if type(total_rows) is not int or total_rows < 1 or rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows):
        raise ValueError('Image alignment requires nonempty integer row indices')
    if np.any((rows < 0) | (rows >= total_rows)) or len(np.unique(rows)) != len(rows):
        raise ValueError('Image row indices must be distinct and inside the metadata')
    try:
        with open(path, 'rb') as stream:
            values = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError, ValueError, RuntimeError, AttributeError) as error:
        raise ValueError('Cannot read image features: ' + str(path)) from error
    if isinstance(values, dict):
        if type(values.get('version')) is not int or values['version'] != 1 or metadata is None or values.get('metadata_hash') != metadata_hash(metadata):
            raise ValueError('Image cache belongs to different split metadata; rebuild the cache')
        if view is not None and values.get('view') != view:
            raise ValueError('Image cache view differs from the requested preprocessing; rebuild the cache')
        values = values.get('images')
    elif require_binding:
        raise ValueError('This dataset requires metadata-bound image caches; run utils/extract.py')
    if not isinstance(values, torch.Tensor) or values.ndim != 4 or values.shape[1] != 3 or min(values.shape[2:]) < 1:
        raise ValueError('Image cache must contain a [rows, 3, height, width] tensor')
    if values.is_complex() or not values.is_floating_point():
        raise ValueError('Image cache must contain real floating-point values')
    # Match cached images to rows retained after text filtering.
    selected = torch.as_tensor(rows, dtype=torch.long)
    if len(values) == total_rows:
        if not torch.equal(selected, torch.arange(total_rows)):
            values = values.index_select(0, selected.to(values.device))
    elif require_binding or len(values) != len(selected):
        raise ValueError('Image cache row count differs from the metadata')
    for chunk in values.split(64):
        if not torch.isfinite(chunk).all():
            raise ValueError('Image cache contains nonfinite values')
    return values

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
