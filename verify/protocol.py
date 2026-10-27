# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

import ast
import io
import logging
from pathlib import Path
import pickle
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import torch
from utils.features import save_features, read_features
from utils.official import verify_files
from utils.partition import paper_split
from utils.utils import Recorder, Averager
from model.training import save_state
from verify.boundary import workspace

class ProtocolTests(unittest.TestCase):
    def test_resnet_keeps_its_pretrained_weights_with_a_relative_cache(self):
        source = ast.parse(Path('model/layers.py').read_text(encoding='utf-8'))
        node = next(item for item in source.body if isinstance(item, ast.ClassDef) and item.name == 'image_extractor')
        backbone = torch.nn.Module()
        for name in ('conv1', 'bn1', 'relu', 'layer1', 'layer2', 'layer3', 'layer4'):
            setattr(backbone, name, torch.nn.Identity())
        backbone.inplanes = 4
        backbone.load_state_dict = Mock()
        parameters = {'pretrained': torch.ones(1)}
        weights = SimpleNamespace(get_state_dict=Mock(return_value=parameters))
        constructor = Mock(return_value=backbone)
        namespace = {'torch': torch, 'F': torch.nn.functional, 'resnet18': constructor, 'ResNet18_Weights': SimpleNamespace(IMAGENET1K_V1=weights)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'layers.py', 'exec'), namespace)
        namespace['image_extractor'](out_channels=4)
        constructor.assert_called_once_with(weights=None)
        backbone.load_state_dict.assert_called_once_with(parameters)
        options = weights.get_state_dict.call_args.kwargs
        self.assertFalse(Path(options['model_dir']).is_absolute())
        self.assertTrue(options['model_dir'].startswith('./model_weights/'))
        self.assertTrue(options['check_hash'])

    def test_remote_resume_uses_a_relative_cache_without_global_changes(self):
        from utils.misc import load_model
        model = Mock()
        state = {'model': {'weight': torch.ones(1)}}
        args = SimpleNamespace(resume='https://example.test/checkpoint.pth', eval=True)
        previous = torch.hub.get_dir()
        with patch('utils.misc.torch.hub.load_state_dict_from_url', return_value=state) as download, redirect_stdout(io.StringIO()):
            load_model(args, model)
        model.load_state_dict.assert_called_once_with(state['model'])
        self.assertFalse(Path(download.call_args.kwargs['model_dir']).is_absolute())
        self.assertTrue(download.call_args.kwargs['model_dir'].startswith('./model_weights/'))
        self.assertEqual(torch.hub.get_dir(), previous)

    def test_paper_defaults_reach_each_dataset_with_fixed_training_seeds(self):
        import main
        from run import Run
        filenames = {
            'weibo': ('train_origin.csv', 'val_origin.csv', 'test_origin.csv'),
            'weibo21': ('train_datasets.xlsx', 'val_datasets.xlsx', 'test_datasets.xlsx'),
            'finefake': ('train.csv', 'val.csv', 'test.csv')
        }
        for dataset, expected in filenames.items():
            for seed in (42, 52, 3047, 3407):
                with self.subTest(dataset=dataset, seed=seed):
                    arguments = ['main.py', '--dataset', dataset.upper(), '--seed', str(seed)]
                    with patch.object(sys, 'argv', arguments), patch.object(main, 'Run') as launch, redirect_stdout(io.StringIO()):
                        main.main()
                    config = launch.call_args.kwargs['config']
                    self.assertEqual((config['max_len'], config['weight_decay'], config['seed']), (128, 0.01, seed))
                    runner = Run(config)
                    self.assertEqual(tuple(Path(path).name for path in (runner.train_path, runner.val_path, runner.test_path)), expected)
                    launch.return_value.main.assert_called_once_with()

    def test_token_fusion_preserves_modalities_at_different_text_lengths(self):
        source = ast.parse(Path('model/layers.py').read_text(encoding='utf-8'))
        node = next(item for item in source.body if isinstance(item, ast.FunctionDef) and item.name == 'concat_tokens')
        namespace = {'torch': torch, 'F': torch.nn.functional}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'layers.py', 'exec'), namespace)
        fuse = namespace['concat_tokens']
        for image_length, text_length in ((197, 128), (197, 197), (128, 197)):
            with self.subTest(image=image_length, text=text_length):
                image = torch.randn(2, image_length, 4, requires_grad=True)
                text = torch.randn(2, text_length, 4, requires_grad=True)
                output = fuse(image, text)
                self.assertEqual(output.shape, (2, max(image_length, text_length), 8))
                torch.testing.assert_close(output[:, :image_length, :4], image, rtol=0, atol=0)
                torch.testing.assert_close(output[:, :text_length, 4:], text, rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(output[:, image_length:, :4]).item(), 0)
                self.assertEqual(torch.count_nonzero(output[:, text_length:, 4:]).item(), 0)
                output.sum().backward()
                torch.testing.assert_close(image.grad, torch.ones_like(image))
                torch.testing.assert_close(text.grad, torch.ones_like(text))
        with self.assertRaisesRegex(ValueError, 'aligned batches'):
            fuse(torch.zeros(2, 197, 4), torch.zeros(3, 128, 4))

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
