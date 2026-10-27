# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

import ast
import io
import logging
import pickle
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import pandas as pd
import torch
from utils.features import save_features, read_features
from utils.official import verify_files
from utils.partition import paper_split
from utils.utils import Recorder, Averager
from model.training import save_state
from verify.boundary import workspace

class ProtocolTests(unittest.TestCase):
    def test_finefake_ratio_and_membership_are_fixed_across_training_seeds(self):
        frame = pd.DataFrame({'text': ['article' + str(index) for index in range(100)], 'label': [index % 2 for index in range(100)]})
        first = paper_split(frame)
        second = paper_split(frame.sample(frac=1, random_state=42).sort_index())
        self.assertEqual([len(first[name]) for name in ('train', 'val', 'test')], [60, 20, 20])
        for name in first:
            self.assertEqual(first[name].index.tolist(), second[name].index.tolist())

    def test_weibo_checks_released_post_ids_for_every_partition(self):
        with workspace() as root:
            paths = {}
            for name, reference in [('train', 'train_id'), ('val', 'validate_id'), ('test', 'test_id')]:
                paths[name] = root / (name + '.csv')
                pd.DataFrame({'post_id': [name], 'content': [name], 'label': [0]}).to_csv(paths[name], index=False)
                (root / (reference + '.pickle')).write_bytes(pickle.dumps([name]))
            self.assertEqual(set(verify_files('weibo', paths, root)), set(paths))
            pd.DataFrame({'post_id': ['train'], 'content': ['train'], 'label': [0]}).to_csv(paths['test'], index=False)
            with self.assertRaisesRegex(ValueError, 'released test'):
                verify_files('weibo', paths, root)

    def test_weibo21_checks_the_released_annotation_partition(self):
        with workspace() as root:
            paths = {}
            for name in ('train', 'val', 'test'):
                frame = pd.DataFrame({'content': [name], 'label': [0]})
                paths[name] = root / (name + '.csv')
                frame.to_csv(paths[name], index=False)
                frame.to_pickle(root / (name + '.pkl'))
            verify_files('weibo21', paths, root)
            pd.DataFrame({'content': ['train'], 'label': [0]}).to_csv(paths['val'], index=False)
            with self.assertRaisesRegex(ValueError, 'released val'):
                verify_files('weibo21', paths, root)

    def test_released_reference_files_are_required(self):
        with workspace() as root, self.assertRaisesRegex(FileNotFoundError, 'reference partition'):
            verify_files('weibo', {'train': root / 'train.csv'}, root)

    def test_bound_caches_require_the_requested_preprocessing_view(self):
        with workspace() as root:
            metadata, cache = root / 'train.csv', root / 'cache.pkl'
            metadata.write_text('text,label\nfirst,0\n', encoding='utf-8')
            save_features(cache, torch.zeros(1, 3, 2, 2), metadata, view='mae')
            with self.assertRaisesRegex(ValueError, 'view'):
                read_features(cache, [0], 1, metadata=metadata, require_binding=True, view='clip')

    def test_bound_caches_keep_all_original_annotation_rows(self):
        with workspace() as root:
            metadata, cache = root / 'train.csv', root / 'cache.pkl'
            metadata.write_text('text,label\nfirst,0\n,1\nthird,0\n', encoding='utf-8')
            save_features(cache, torch.zeros(2, 3, 2, 2), metadata)
            with self.assertRaisesRegex(ValueError, 'row count'):
                read_features(cache, [0, 2], 3, metadata=metadata, require_binding=True)

    def gossip(self, root, validation):
        source = ast.parse(__import__('pathlib').Path('model/gossip.py').read_text(encoding='utf-8'))
        node = next(item for item in source.body if isinstance(item, ast.ClassDef) and item.name == 'Trainer')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.1))

            def forward(self, label, **kwargs):
                return (self.value.expand_as(label),) * 4

        class Progress:
            def __init__(self, items):
                self.items = items

            def __iter__(self):
                return iter(self.items)

            def set_description(self, *args):
                pass

            def set_postfix(self, **kwargs):
                pass

        namespace = {'torch': torch, 'os': __import__('os'), 'logger': logging.getLogger('protocol'), 'Recorder': Recorder, 'Averager': Averager, 'save_state': save_state, 'clipdata2gpu': lambda batch: batch, 'tqdm': SimpleNamespace(tqdm=lambda items, **kwargs: Progress(items))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'gossip.py', 'exec'), namespace)
        trainer = object.__new__(namespace['Trainer'])
        trainer.model = Model()
        trainer.lr, trainer.weight_decay, trainer.early_stop, trainer.epoches = 0.001, 0, 1, 1
        trainer.train_loader, trainer.val_loader, trainer.test_loader = [{'label': torch.tensor([0., 1.])}], ['val'], ['test']
        trainer.save_param_dir, trainer.category_dict, trainer.metric_key_for_early_stop = str(root), {}, 'metric'
        trainer.test = validation
        return trainer

    def test_failed_validation_never_reuses_a_stale_checkpoint(self):
        with workspace() as root:
            (root / 'best_model.pth').write_bytes(b'old checkpoint')
            trainer = self.gossip(root, lambda loader: (_ for _ in ()).throw(ValueError('failed validation')))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'Validation failed'):
                trainer.train()
            self.assertEqual((root / 'best_model.pth').read_bytes(), b'old checkpoint')

    def test_gossip_tests_only_after_validation_selects_a_checkpoint(self):
        with workspace() as root:
            calls = []
            def validate(loader):
                calls.append(loader[0])
                return {'metric': 0.0, 'acc': 0.0}
            trainer = self.gossip(root, validate)
            with redirect_stdout(io.StringIO()):
                trainer.train()
            self.assertEqual(calls, ['val', 'test'])
            self.assertTrue((root / 'best_model.pth').is_file())

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
