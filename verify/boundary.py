# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import pickle
import tempfile
import subprocess
import sys
import importlib
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from model.training import FGM, EMA, WarmupCosineAnnealingLR, save_state
from utils.features import binary_labels, read_features, clip_tokens, save_features
from utils.partition import grouped_split, connected_rows, audit_frames
from utils.utils import calculate_metrics, metrics, metricsTrueFalse, Recorder
from model.net import Trainer, DDIN, AdaptiveContrastiveLoss

class TinyModel(torch.nn.Module):
    def __init__(self, fail_adversarial=False):
        super().__init__()
        self.bert = torch.nn.Module()
        self.bert.embeddings = torch.nn.Module()
        self.bert.embeddings.word_embeddings = torch.nn.Embedding(4, 4)
        self.classifier = torch.nn.Linear(4, 2)
        self.calls, self.fail_adversarial = 0, fail_adversarial

    def forward(self, content, **kwargs):
        self.calls += 1
        if self.fail_adversarial and self.calls == 2:
            raise RuntimeError('adversarial inference failure')
        features = self.bert.embeddings.word_embeddings(content).mean(dim=1)
        logits = self.classifier(features)
        return (logits,) * 4 + (features,) * 4

@contextmanager
def workspace():
    with tempfile.TemporaryDirectory(prefix='test-', dir='.') as directory:
        root = Path(directory)
        yield root.relative_to(Path.cwd()) if root.is_absolute() else root

class MetricTests(unittest.TestCase):
    def test_default_categories_and_probability_ties(self):
        result = calculate_metrics([0, 1], [0.1, 0.5])
        self.assertEqual(result['acc'], 1)
        self.assertEqual(result['all']['auc'], 1)

    def test_category_auc_survives_classification_statistics(self):
        result = metrics([0, 1, 0], [0.1, 0.9, 0.2], [0, 0, 1], {'first': 0, 'second': 1})
        self.assertEqual(result['first']['auc'], 1)
        self.assertIsNone(result['second']['auc'])

    def test_single_class_auc_is_explicitly_unavailable(self):
        result = calculate_metrics([1, 1], [0.9, 0.8])
        self.assertIsNone(result['auc'])
        self.assertEqual(result['metric'], 1)

    def test_class_f1_uses_the_actual_precision_recall_sum(self):
        result = metricsTrueFalse([1, 1, 1, 0, 0], [0.9, 0.1, 0.1, 0.9, 0.9], [0] * 5, {'all': 0})
        self.assertAlmostEqual(result['fake']['F1'], 1 / 3)

    def test_invalid_metric_arrays_are_rejected(self):
        for labels, probabilities in (([], []), ([0, 1], [0.5]), ([0.2], [0.5]), ([0], [float('nan')]), ([1], [1.1]), ([1], [0.1j])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                calculate_metrics(labels, probabilities)

    def test_unknown_or_duplicate_category_ids_are_rejected(self):
        for categories, mapping in (([1], {'first': 0}), ([0], {'first': 0, 'second': 0}), ([], {'first': 0})):
            with self.assertRaises(ValueError):
                metrics([0], [0.1], categories, mapping)

    def test_first_zero_epoch_is_saved_and_patience_is_respected(self):
        recorder = Recorder(2)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(recorder.add({'metric': 0.0}), 'save')
            self.assertEqual(recorder.add({'metric': 0.0}), 'continue')
            self.assertEqual(recorder.add({'metric': 0.0}), 'esc')

    def test_recorder_retains_an_independent_snapshot(self):
        recorder = Recorder(2)
        result = {'metric': 0.5, 'nested': {'value': 1}}
        with redirect_stdout(io.StringIO()):
            recorder.add(result)
        result['nested']['value'] = 99
        self.assertEqual(recorder.max['nested']['value'], 1)

    def test_invalid_validation_scores_do_not_advance_the_recorder(self):
        recorder = Recorder(2)
        for value in (float('nan'), float('inf'), -1, 2, True):
            with self.assertRaises(ValueError):
                recorder.add({'metric': value})
        self.assertEqual(recorder.curindex, 0)

    def test_invalid_patience_is_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                Recorder(value)

class TrainingTests(unittest.TestCase):
    def test_fgm_restores_after_a_failed_perturbation_update(self):
        model, weight = self.embedding_model()
        original = weight.detach().clone()
        weight.grad = torch.ones_like(weight)
        attack = FGM(model, epsilon=1e300)
        with self.assertRaises((RuntimeError, OverflowError)):
            attack.attack()
        self.assertTrue(torch.equal(weight, original))
        self.assertFalse(attack.backup)

    def test_incomplete_ema_state_is_rejected_before_weight_changes(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
        average = EMA(model)
        original = {name: value.detach().clone() for name, value in model.state_dict().items()}
        first, last = list(average.shadow)[0], list(average.shadow)[-1]
        average.shadow[first].zero_()
        del average.shadow[last]
        with self.assertRaises(RuntimeError):
            average.apply_shadow()
        self.assertTrue(all(torch.equal(value, original[name]) for name, value in model.state_dict().items()))
        self.assertFalse(average.backup)

    def trainer(self, model, directory):
        batch = {'content': torch.tensor([[1, 2], [2, 3]]), 'label': torch.tensor([0, 1]), 'category': torch.zeros(2, dtype=torch.long)}
        with patch('model.net.DDIN', return_value=model):
            return Trainer(768, [4], 'unused', False, 0.001, 0.1, [batch], [batch], [batch], {'all': 0}, 0.0, str(directory), 2, 1)

    def test_one_epoch_training_saves_a_zero_f1_checkpoint(self):
        with workspace() as root:
            trainer = self.trainer(TinyModel(), root)
            result = ({'metric': 0.0},) * 4
            with patch.object(trainer, 'test', return_value=result), patch('model.net.tqdm.tqdm', side_effect=lambda values: values), redirect_stdout(io.StringIO()):
                self.assertEqual(trainer.train(), result)
            state = torch.load(root / 'parameter_DDIN.pkl', map_location='cpu', weights_only=True)
            self.assertEqual(set(state), set(trainer.model.state_dict()))

    def test_trainer_restores_embeddings_after_adversarial_failure(self):
        model = TinyModel(fail_adversarial=True)
        original = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with workspace() as root:
            trainer = self.trainer(model, root)
            with patch('model.net.tqdm.tqdm', side_effect=lambda values: values), self.assertRaisesRegex(RuntimeError, 'adversarial inference failure'):
                trainer.train()
            self.assertTrue(all(torch.equal(value, original[name]) for name, value in model.state_dict().items()))
            self.assertFalse((root / 'parameter_DDIN.pkl').exists())

    def test_contrastive_loss_handles_a_single_sample(self):
        features = torch.ones(1, 4, requires_grad=True)
        loss = AdaptiveContrastiveLoss()(features, features, torch.tensor([0]))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_invalid_contrastive_temperatures_are_rejected(self):
        for value in (0, -1, True, float('nan')):
            with self.assertRaises(ValueError):
                AdaptiveContrastiveLoss(value)

    def embedding_model(self):
        model = torch.nn.Module()
        model.bert = torch.nn.Module()
        model.bert.embeddings = torch.nn.Module()
        model.bert.embeddings.word_embeddings = torch.nn.Embedding(4, 2)
        return model, model.bert.embeddings.word_embeddings.weight

    def test_fgm_skips_missing_and_zero_gradients(self):
        model, weight = self.embedding_model()
        before = weight.detach().clone()
        attack = FGM(model)
        attack.attack()
        attack.restore()
        weight.grad = torch.zeros_like(weight)
        attack.attack()
        attack.restore()
        self.assertTrue(torch.equal(weight, before))

    def test_fgm_restores_weights_after_adversarial_failure(self):
        model, weight = self.embedding_model()
        before = weight.detach().clone()
        weight.grad = torch.ones_like(weight)
        attack = FGM(model)
        with self.assertRaisesRegex(RuntimeError, 'inference failure'):
            with attack.perturb():
                self.assertAlmostEqual(float(torch.norm(weight.detach() - before)), 0.5, places=5)
                raise RuntimeError('inference failure')
        self.assertTrue(torch.equal(weight, before))
        self.assertFalse(attack.backup)

    def test_fgm_rejects_nonfinite_gradients_before_mutation(self):
        model, weight = self.embedding_model()
        before = weight.detach().clone()
        weight.grad = torch.full_like(weight, float('inf'))
        with self.assertRaises(FloatingPointError):
            FGM(model).attack()
        self.assertTrue(torch.equal(weight, before))

    def test_ema_restores_weights_after_validation_failure(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        average = EMA(model, 0.5)
        with torch.no_grad():
            model.weight.fill_(2)
        average.update()
        with self.assertRaisesRegex(RuntimeError, 'validation failure'):
            with average.average_parameters():
                self.assertEqual(model.weight.item(), 1)
                raise RuntimeError('validation failure')
        self.assertEqual(model.weight.item(), 2)

    def test_nested_ema_application_does_not_lose_live_weights(self):
        model = torch.nn.Linear(1, 1, bias=False)
        average = EMA(model)
        original = model.weight.detach().clone()
        with average.average_parameters():
            with self.assertRaises(RuntimeError):
                average.apply_shadow()
            with self.assertRaises(RuntimeError):
                average.update()
        self.assertTrue(torch.equal(model.weight, original))

    def test_scheduler_handles_one_epoch_and_holds_its_final_rate(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
        scheduler = WarmupCosineAnnealingLR(optimizer, 1, 1)
        for _ in range(3):
            scheduler.step()
            self.assertEqual(optimizer.param_groups[0]['lr'], 0.1)

    def test_cosine_schedule_never_rebounds_after_training(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
        scheduler = WarmupCosineAnnealingLR(optimizer, 0, 2, eta_min=0.01)
        scheduler.step()
        for _ in range(3):
            scheduler.step()
            self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)

    def test_invalid_training_helper_parameters_are_rejected(self):
        model = torch.nn.Linear(1, 1)
        for value in (-1, float('nan'), True):
            with self.assertRaises(ValueError):
                FGM(model, value)
            with self.assertRaises(ValueError):
                EMA(model, value)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for warmup, maximum in ((3, 1), (0, 0), (-1, 3), (True, 3)):
            with self.assertRaises(ValueError):
                WarmupCosineAnnealingLR(optimizer, warmup, maximum)

    def test_checkpoint_roundtrip_contains_only_model_tensors(self):
        model = torch.nn.Linear(2, 1)
        with workspace() as root:
            path = root / 'model.pkl'
            save_state(model, path)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
            self.assertTrue(torch.equal(loaded['weight'], model.weight))

    def test_failed_save_preserves_the_previous_checkpoint(self):
        with workspace() as root:
            path = root / 'model.pkl'
            path.write_bytes(b'previous checkpoint')
            def fail(state, temporary):
                temporary.write(b'partial checkpoint')
                raise OSError('disk failure')
            with patch('model.training.torch.save', side_effect=fail), self.assertRaises(OSError):
                save_state(torch.nn.Linear(1, 1), path)
            self.assertEqual(path.read_bytes(), b'previous checkpoint')
            self.assertEqual(list(root.iterdir()), [path])

    def test_nonfinite_weights_do_not_replace_a_checkpoint(self):
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(float('nan'))
        with workspace() as root:
            path = root / 'model.pkl'
            path.write_bytes(b'previous')
            with self.assertRaises(FloatingPointError):
                save_state(model, path)
            self.assertEqual(path.read_bytes(), b'previous')

class FeatureTests(unittest.TestCase):
    def test_image_preprocessing_writes_bound_caches_under_a_custom_root(self):
        from PIL import Image
        from torchvision.transforms import ToTensor
        modules = {name: ModuleType(name) for name in ('cn_clip', 'cn_clip.clip', 'cn_clip.clip.utils')}
        modules['cn_clip.clip.utils'].image_transform = lambda size: ToTensor()
        with patch.dict(sys.modules, modules):
            extraction = importlib.import_module('utils.extract')
        with workspace() as root:
            Image.new('RGB', (2, 2), (100, 50, 20)).save(root / 'photo.png')
            metadata = root / 'train.csv'
            metadata.write_text('image_path,label\nphoto.png,0\n', encoding='utf-8')
            with patch.object(extraction, 'tqdm', side_effect=lambda values, **kwargs: values), redirect_stdout(io.StringIO()):
                extraction.encode_split('train', ToTensor(), ToTensor(), root=root)
            for name in ('f_train_loader.pkl', 'f_train_clip.pkl'):
                images = read_features(root / name, [0], 1, metadata=metadata, require_binding=True)
                self.assertEqual(tuple(images.shape), (1, 3, 2, 2))

    def test_finefake_loader_keeps_images_aligned_after_filtering(self):
        import pandas as pd
        parent, module = ModuleType('cn_clip'), ModuleType('cn_clip.clip')
        module.tokenize = lambda texts: torch.ones(len(texts), 52, dtype=torch.long)
        parent.clip = module
        with patch.dict(sys.modules, {'cn_clip': parent, 'cn_clip.clip': module}):
            loader_module = importlib.import_module('utils.fld')
        with workspace() as root:
            metadata = root / 'data.csv'
            pd.DataFrame({'text': ['first', None, 'third'], 'label': [0, 1, 1], 'category': ['Uncategorized'] * 3}).to_csv(metadata, index=False)
            images = root / 'images.pkl'
            values = torch.arange(3, dtype=torch.float32)[:, None, None, None].expand(3, 3, 2, 2)
            save_features(images, values, metadata)
            tokenizer = lambda texts, *args: (torch.ones(len(texts), 4, dtype=torch.long), torch.ones(len(texts), 4))
            loader = loader_module.bert_data(4, 2, 'unused', {'Uncategorized': 0}, num_workers=0)
            with patch.object(loader_module, 'word2input', side_effect=tokenizer), redirect_stdout(io.StringIO()):
                result = loader.load_data(metadata, images, images, False)
            self.assertEqual(result.dataset.tensors[4][:, 0, 0, 0].tolist(), [0, 2])
            self.assertEqual(result.dataset.tensors[6].shape, (2, 52))

    def test_clip_fallback_uses_the_tokenizers_context_length(self):
        def tokenize(texts):
            if 'broken' in texts:
                raise ValueError('invalid text')
            return torch.ones(len(texts), 52, dtype=torch.long)
        tokens = clip_tokens(['valid', 'broken'], tokenize)
        self.assertEqual(tuple(tokens.shape), (2, 52))

    def test_failed_fallback_tokenization_is_reported_cleanly(self):
        with self.assertRaisesRegex(ValueError, 'fallback text'):
            clip_tokens(['broken'], lambda texts: (_ for _ in ()).throw(ValueError('tokenizer failure')))

    def test_invalid_token_matrices_are_rejected(self):
        for tokens in (torch.zeros(1, 52), torch.zeros(2, 52, dtype=torch.long), torch.full((1, 52), -1, dtype=torch.long)):
            with self.assertRaises(ValueError):
                clip_tokens(['valid'], lambda texts: tokens)

    def test_filtered_text_rows_keep_the_corresponding_images(self):
        values = torch.arange(3, dtype=torch.float32)[:, None, None, None].expand(3, 3, 2, 2)
        with workspace() as root:
            path = root / 'images.pkl'
            path.write_bytes(pickle.dumps(values))
            selected = read_features(path, [0, 2], 3)
            self.assertEqual(selected[:, 0, 0, 0].tolist(), [0, 2])

    def test_prefiltered_cache_is_accepted(self):
        values = torch.zeros(2, 3, 2, 2)
        with workspace() as root:
            path = root / 'images.pkl'
            path.write_bytes(pickle.dumps(values))
            self.assertEqual(read_features(path, [0, 2], 3).shape[0], 2)

    def test_metadata_bound_cache_rejects_a_changed_split(self):
        with workspace() as root:
            metadata, cache = root / 'rows.csv', root / 'images.pkl'
            metadata.write_text('text,label\nfirst,0\n', encoding='utf-8')
            save_features(cache, torch.ones(1, 3, 2, 2), metadata)
            self.assertEqual(read_features(cache, [0], 1, metadata=metadata, require_binding=True).shape[0], 1)
            metadata.write_text('text,label\nchanged,1\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'different split metadata'):
                read_features(cache, [0], 1, metadata=metadata, require_binding=True)

    def test_finefake_rejects_unbound_caches(self):
        with workspace() as root:
            cache = root / 'images.pkl'
            cache.write_bytes(pickle.dumps(torch.ones(1, 3, 2, 2)))
            with self.assertRaisesRegex(ValueError, 'metadata-bound'):
                read_features(cache, [0], 1, require_binding=True)

    def test_failed_feature_save_preserves_the_original_cache(self):
        with workspace() as root:
            metadata, cache = root / 'rows.csv', root / 'images.pkl'
            metadata.write_text('text,label\nfirst,0\n', encoding='utf-8')
            cache.write_bytes(b'previous cache')
            with patch('utils.features.pickle.dump', side_effect=OSError('disk failure')), self.assertRaises(OSError):
                save_features(cache, torch.ones(1, 3, 2, 2), metadata)
            self.assertEqual(cache.read_bytes(), b'previous cache')
            self.assertEqual(set(root.iterdir()), {metadata, cache})

    def test_invalid_cache_shapes_counts_and_values_are_rejected(self):
        for values in (torch.zeros(4, 3, 2, 2), torch.zeros(2, 2), torch.full((2, 3, 2, 2), float('nan')), [1, 2]):
            with workspace() as root:
                path = root / 'images.pkl'
                path.write_bytes(pickle.dumps(values))
                with self.assertRaises(ValueError):
                    read_features(path, [0, 2], 3)

    def test_corrupt_caches_are_reported_cleanly(self):
        with workspace() as root:
            path = root / 'images.pkl'
            path.write_bytes(b'not a pickle')
            with self.assertRaisesRegex(ValueError, 'Cannot read image features'):
                read_features(path, [0], 1)

    def test_invalid_row_indices_are_rejected_before_reading(self):
        for indices in ([], [0, 0], [-1], [3], [0.5]):
            with self.assertRaises(ValueError):
                read_features('missing.pkl', indices, 3)

    def test_fractional_and_nonfinite_labels_are_rejected(self):
        for labels in ([0.5], [float('nan')], [2], ['0']):
            with self.assertRaises(ValueError):
                binary_labels(labels)
        self.assertEqual(binary_labels([0, 1]).tolist(), [0, 1])

class InterfaceTests(unittest.TestCase):
    def test_custom_relative_dataset_root_is_used(self):
        from run import Run
        config = dict.fromkeys(('use_cuda', 'lr', 'batchsize', 'emb_dim', 'max_len', 'num_workers', 'vocab_file', 'early_stop', 'bert', 'seed', 'weight_decay', 'epoch', 'save_param_dir'))
        config.update({'model_name': 'DDIN', 'emb_type': 'bert', 'dataset': 'weibo', 'root_path': 'datasets/custom', 'model': {'mlp': {'dims': [4], 'dropout': 0.1}}})
        runner = Run(config)
        self.assertEqual(Path(runner.train_path), Path('datasets/custom/train_origin.csv'))
    def test_split_path_aliases_are_rejected(self):
        from run import Run
        config = dict.fromkeys(('use_cuda', 'lr', 'batchsize', 'emb_dim', 'max_len', 'num_workers', 'vocab_file', 'early_stop', 'bert', 'root_path', 'seed', 'weight_decay', 'epoch', 'save_param_dir'))
        config.update({'model_name': 'DDIN', 'emb_type': 'bert', 'dataset': 'FineFake', 'model': {'mlp': {'dims': [4], 'dropout': 0.1}}, 'finefake_val': 'test.csv', 'finefake_test': 'sub/../test.csv'})
        with self.assertRaisesRegex(ValueError, 'distinct'):
            Run(config)

    def test_unsupported_driver_configuration_fails_cleanly(self):
        from run import Run
        for config in ({'dataset': 'unknown'}, {'dataset': 'weibo', 'emb_type': 'w2v'}, {'dataset': 'weibo', 'emb_type': 'bert', 'model_name': 'unknown'}):
            with self.assertRaises(ValueError):
                Run(config)

    def test_help_does_not_load_optional_backbones(self):
        result = subprocess.run([sys.executable, '-B', 'main.py', '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--dataset', result.stdout)

    def test_invalid_cli_values_fail_before_loading_data(self):
        for arguments in (['--epoch', '0'], ['--lr', 'nan'], ['--emb_type', 'w2v'], ['--seed', '-1']):
            result = subprocess.run([sys.executable, '-B', 'main.py', *arguments], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('error:', result.stderr)

    def test_forward_supports_custom_width_and_half_precision_clip_embeddings(self):
        class TextBackbone(torch.nn.Module):
            def forward(self, content, **kwargs):
                return SimpleNamespace(last_hidden_state=torch.ones(*content.shape, 768))
        class ImageBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
            def forward_encoder(self, image, **kwargs):
                return (torch.ones(len(image), 2, 768),)
        class ClipBackbone(torch.nn.Module):
            def encode_text(self, values):
                return torch.ones(len(values), 512, dtype=torch.float16)
            def encode_image(self, values):
                return torch.ones(len(values), 512, dtype=torch.float16)
        modules = {name: ModuleType(name) for name in ('mae', 'transformers', 'timm', 'timm.models', 'timm.models.vision_transformer', 'cn_clip', 'cn_clip.clip')}
        modules['mae'].mae_vit_base_patch16 = lambda **kwargs: ImageBackbone()
        modules['transformers'].BertModel = SimpleNamespace(from_pretrained=lambda path: TextBackbone())
        modules['timm.models.vision_transformer'].Block = lambda **kwargs: torch.nn.Identity()
        with patch.dict(sys.modules, modules), patch('model.net.torch.load', return_value={'model': {}}), patch.object(modules['cn_clip.clip'], 'load_from_name', create=True, return_value=(ClipBackbone(), None)) as loader:
            model = DDIN('unused', 'weights/mae.pt', 'ViT-B-16', num_domains=1, hidden_dim=8).eval()
        with torch.no_grad():
            result = model(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3), torch.zeros(1, 3, 4, 4), torch.zeros(1, dtype=torch.long))
        self.assertEqual(result[0].shape, (1, 2))
        self.assertTrue(torch.isfinite(result[0]).all())
        self.assertEqual(loader.call_args.kwargs['device'], 'cpu')
        self.assertFalse(any(value.requires_grad for value in model.mae.parameters()))

class PartitionTests(unittest.TestCase):
    def frame(self, count=30):
        import pandas as pd
        return pd.DataFrame({'text': ['article' + str(index) for index in range(count)], 'label': [index % 2 for index in range(count)], 'image_path': ['images/' + str(index) + '.jpg' for index in range(count)]})

    def test_grouped_split_is_reproducible_and_transitively_disjoint(self):
        frame = self.frame()
        frame.loc[1, 'text'] = frame.loc[0, 'text']
        frame.loc[2, 'image_path'] = frame.loc[1, 'image_path']
        first, second = grouped_split(frame, seed=42), grouped_split(frame, seed=42)
        for name in first:
            self.assertEqual(first[name].index.tolist(), second[name].index.tolist())
        self.assertEqual(sum(0 in part.index and 1 in part.index and 2 in part.index for part in first.values()), 1)
        self.assertEqual(sum(audit_frames(first).values()), len(frame))

    def test_normalized_text_leakage_is_rejected(self):
        frames = grouped_split(self.frame())
        frames['train'].iloc[0, 0] = 'A &nbsp; B'
        frames['test'].iloc[0, 0] = 'ab'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            audit_frames(frames)

    def test_shared_source_ids_are_kept_in_one_component(self):
        frame = self.frame(3)
        frame['source_id'] = ['source', 'source', 'another']
        self.assertEqual(sorted(map(len, connected_rows(frame))), [1, 2])

    def test_copied_images_are_detected_by_content(self):
        frames = grouped_split(self.frame())
        with workspace() as root:
            first, second = root / 'first.jpg', root / 'second.jpg'
            first.write_bytes(b'identical image data')
            second.write_bytes(first.read_bytes())
            frames['train'].iloc[0, frames['train'].columns.get_loc('image_path')] = 'first.jpg'
            frames['test'].iloc[0, frames['test'].columns.get_loc('image_path')] = 'second.jpg'
            with self.assertRaisesRegex(ValueError, 'leakage'):
                audit_frames(frames, image_root=root)

    def test_grouping_failure_does_not_write_split_files(self):
        from utils.fsplit import prepare
        frame = self.frame(3)
        frame['text'] = 'one original article'
        with workspace() as root:
            frame.to_pickle(root / 'FineFake.pkl')
            previous = root / 'train.csv'
            previous.write_bytes(b'previous split')
            with self.assertRaisesRegex(ValueError, 'independent'):
                prepare(root)
            self.assertEqual(previous.read_bytes(), b'previous split')
            self.assertFalse((root / 'test.csv').exists())

    def test_prepare_writes_audited_grouped_splits(self):
        from utils.fsplit import prepare
        from utils.partition import audit_files
        with workspace() as root:
            self.frame().to_pickle(root / 'FineFake.pkl')
            result = prepare(root)
            self.assertEqual(sum(len(part) for part in result.values()), 30)
            self.assertEqual(sum(audit_files({name: root / (name + '.csv') for name in result}).values()), 30)

    def test_training_audits_splits_before_loading_backbones(self):
        from run import Run
        runner = object.__new__(Run)
        runner.train_path, runner.val_path, runner.test_path, runner.root_path = 'train.csv', 'val.csv', 'test.csv', '.'
        with patch('utils.partition.audit_files', side_effect=ValueError('split leakage')), patch.object(runner, 'get_dataloader') as loader:
            with self.assertRaisesRegex(ValueError, 'split leakage'):
                runner.main()
            loader.assert_not_called()

if __name__ == '__main__':
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
