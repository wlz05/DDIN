# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
from copy import deepcopy
def batch2dict(batch):
    """Convert a list batch into the named dict the trainers expect (CPU-safe).

    List order follows the loaders' TensorDataset:
    content, content_masks, label, category, image, clip_image, clip_text.
    """
    if isinstance(batch, dict):
        return batch
    keys = ('content', 'content_masks', 'label', 'category',
            'image', 'clip_image', 'clip_text')
    return {k: batch[i] for i, k in enumerate(keys) if i < len(batch)}
def clipdata2gpu(batch):
    if isinstance(batch, dict):
        batch_data = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in batch.items()}
        if 'clip_attention_mask' not in batch_data and 'clip_text' in batch_data:
            batch_data['clip_attention_mask'] = (batch_data['clip_text'] != 0).long()
        return batch_data
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image':batch[4].cuda(),
        'clip_image':batch[5].cuda(),
        'clip_text': batch[6].cuda()
    }
    return batch_data
def data2gpu(batch):
    if isinstance(batch, dict):
        batch_data = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in batch.items()}
        return batch_data
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image':batch[4].cuda()
    }
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def metricsTrueFalse(y_true, y_pred, category, category_dict):
    result = metrics(y_true, y_pred, category, category_dict)
    labels, probabilities, _, _ = metric_inputs(y_true, y_pred, category, category_dict)
    predictions = (probabilities >= 0.5).astype(int)
    for name, positive in (('real', 0), ('fake', 1)):
        truth, predicted = labels == positive, predictions == positive
        tp = int(np.sum(truth & predicted))
        precision = tp / max(int(predicted.sum()), 1)
        recall = tp / max(int(truth.sum()), 1)
        total = precision + recall
        result[name] = {'precision': precision, 'recall': recall, 'F1': 2 * precision * recall / total if total else 0.0}
    return result

def metric_inputs(y_true, y_pred, category=None, category_dict=None):
    labels, probabilities = np.asarray(y_true), np.asarray(y_pred)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not labels.size:
        raise ValueError('Metrics require equally sized nonempty vectors')
    for values in (labels, probabilities):
        if values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
            raise ValueError('Metrics require finite numeric inputs')
    if not np.isin(labels, (0, 1)).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Metrics require binary labels and probabilities in [0, 1]')
    if category is None:
        groups, mapping = np.zeros(len(labels), dtype=int), {'all': 0}
    else:
        groups = np.asarray(category)
        if groups.shape != labels.shape or groups.dtype.kind not in 'iu':
            raise ValueError('Categories require an integer ID for every sample')
        mapping = dict(category_dict) if category_dict else {str(value): int(value) for value in np.unique(groups)}
        values = list(mapping.values())
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
            raise ValueError('Category IDs must be integers')
        if len(set(values)) != len(values) or not np.isin(groups, values).all():
            raise ValueError('Category IDs must be unique and cover every sample')
    return labels.astype(int), probabilities.astype(float), groups, mapping

def metrics(y_true, y_pred, category, category_dict):
    labels, probabilities, groups, mapping = metric_inputs(y_true, y_pred, category, category_dict)
    # Use the same threshold for aggregate and class-specific metrics.
    predictions = (probabilities >= 0.5).astype(int)
    result = {'metric': float(f1_score(labels, predictions, average='macro', zero_division=0)),
              'recall': float(recall_score(labels, predictions, average='macro', zero_division=0)),
              'precision': float(precision_score(labels, predictions, average='macro', zero_division=0)),
              'acc': float(accuracy_score(labels, predictions)),
              'auc': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None}
    for name, group in mapping.items():
        mask = groups == group
        if not mask.any():
            continue
        truth, predicted, scores = labels[mask], predictions[mask], probabilities[mask]
        # Preserve AUC alongside the category classification metrics.
        result[name] = {'auc': round(float(roc_auc_score(truth, scores)), 4) if len(np.unique(truth)) == 2 else None,
                        'precision': round(float(precision_score(truth, predicted, average='macro', zero_division=0)), 4),
                        'recall': round(float(recall_score(truth, predicted, average='macro', zero_division=0)), 4),
                        'fscore': round(float(f1_score(truth, predicted, average='macro', zero_division=0)), 4),
                        'acc': round(float(accuracy_score(truth, predicted)), 4)}
    return result

def calculate_metrics(y_true, y_pred, category=None, category_dict=None):
    return metrics(y_true, y_pred, category, category_dict)

class Recorder():

    def __init__(self, early_step):
        if type(early_step) is not int or early_step < 1:
            raise ValueError('Early stopping patience must be a positive integer')
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        if not isinstance(x, dict) or not isinstance(x.get('metric'), (int, float, np.number)) or isinstance(x['metric'], bool) or not np.isfinite(x['metric']) or not 0 <= x['metric'] <= 1:
            raise ValueError('Validation metric must be finite and lie in [0, 1]')
        self.cur = deepcopy(x)
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        # Save the first valid epoch even when its F1 is zero.
        if self.maxindex == 0 or self.cur['metric'] > self.max['metric']:
            self.max = deepcopy(self.cur)
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
