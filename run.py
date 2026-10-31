# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

import os
import ntpath
import json
import math
import random
import statistics
import tempfile
from pathlib import Path
from copy import deepcopy

SEEDS = (42, 52, 3047, 3407)

def seed_summary(records, seeds=SEEDS):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("Seeds must be distinct unsigned 32-bit integers")
    if any(not isinstance(row, dict) or type(row.get("seed")) is not int or "metrics" not in row or not isinstance(row.get("protocol"), dict) for row in records):
        raise ValueError("Every seed result needs metrics and an experiment protocol")
    if len(records) != len(seeds) or {row["seed"] for row in records} != set(seeds):
        raise ValueError("Missing or duplicate seed results; partial averages are forbidden")
    if any(row["protocol"] != records[0]["protocol"] for row in records):
        raise ValueError("All seeds must use the same inputs and experiment settings")
    def flatten(value, prefix=""):
        result = {}
        if isinstance(value, dict):
            for key, item in value.items():
                result.update(flatten(item, prefix + ("." if prefix else "") + str(key)))
        elif value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
            if value is not None and not math.isfinite(value):
                raise ValueError("Seed metrics must be finite or explicitly undefined")
            result[prefix] = value
        return result
    byseed = {str(row["seed"]): row["metrics"] for row in records}
    flat = {seed: flatten(byseed[str(seed)]) for seed in seeds}
    metrics = {}
    for key in sorted(set().union(*(value.keys() for value in flat.values()))):
        values = [flat[seed].get(key) for seed in seeds]
        undefined = [seed for seed, value in zip(seeds, values) if value is None]
        metrics[key] = {"mean": statistics.mean(values) if not undefined else None, "sample_std": statistics.stdev(values) if not undefined and len(values) > 1 else None, "undefined_seeds": undefined}
    return {"seeds": seeds, "seed_count": len(seeds), "byseed": byseed, "metrics": metrics, "protocol": records[0]["protocol"], "aggregation": "Compute each seed's test metrics first, then take their equally weighted arithmetic mean."}

def write_result(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".result-", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def run_seeds(config, seeds=SEEDS):
    import numpy as np
    import torch
    from utils.features import metadata_hash
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("Seeds must be distinct unsigned 32-bit integers")
    root = Path(config["save_param_dir"])
    records, frozen = [], None
    settings = deepcopy(config)
    settings.pop("seed", None)
    settings.pop("save_param_dir", None)
    for seed in seeds:
        current = deepcopy(config)
        current.update(seed=seed, save_param_dir=(root / ("seed" + str(seed))).as_posix())
        runner = Run(current)
        inputs = {path: metadata_hash(path) for path in (runner.train_path, runner.val_path, runner.test_path)}
        if frozen is not None and inputs != frozen:
            raise ValueError("Dataset partitions changed between training seeds")
        frozen = inputs
        targets = [root / "summary.json", Path(current["save_param_dir"]) / current["model_name"] / "result.json"]
        if any(target.resolve() == Path(path).resolve() for target in targets for path in inputs):
            raise ValueError("Seed reports cannot overwrite dataset inputs")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if current.get("use_cuda"):
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        outcome = runner.main()
        metrics = outcome[0] if isinstance(outcome, tuple) else outcome
        if not isinstance(metrics, dict):
            raise ValueError("Training must return its selected-checkpoint test metrics")
        if any(metadata_hash(path) != checksum for path, checksum in inputs.items()):
            raise ValueError("Dataset partitions changed during a training run")
        record = {"seed": seed, "metrics": metrics, "protocol": {"settings": settings, "inputs": inputs}}
        seed_summary([record], [seed])
        write_result(targets[1], record)
        records.append(record)
    result = seed_summary(records, seeds)
    write_result(root / "summary.json", result)
    return result

class Run():
    def __init__(self, config):
        if not isinstance(config.get('dataset'), str) or config['dataset'].lower() not in ('weibo', 'weibo21', 'finefake'):
            raise ValueError('Choose weibo, weibo21, or finefake')
        if config.get('emb_type') != 'bert':
            raise ValueError('The configured loaders require BERT tokenization')
        if config.get('model_name') not in ('DDIN', 'Gossip'):
            raise ValueError('Choose the DDIN or Gossip model driver')
        self.configinfo = dict(config)
        path_keys = ("root_path", "official_root", "vocab_file", "bert", "save_param_dir", "finefake_train", "finefake_val", "finefake_test")
        for key in path_keys:
            value = self.configinfo.get(key)
            if value is None:
                continue
            if not isinstance(value, (str, os.PathLike)):
                raise ValueError("Use a nonempty relative path for " + key)
            value = os.fspath(value)
            if not isinstance(value, str) or not value.strip() or "\x00" in value or value.startswith(("/", "\\")) or ntpath.isabs(value) or ntpath.splitdrive(value)[0] or os.path.isabs(value):
                raise ValueError("Use a nonempty relative path for " + key)
            self.configinfo[key] = value
        config = self.configinfo

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config.get('root_path') or {'weibo': './data/', 'weibo21': './weibo21/', 'finefake': './FineFake/'}[config['dataset'].lower()]
        if os.path.isabs(self.root_path) or os.path.splitdrive(self.root_path)[0]:
            raise ValueError('Dataset root must be a relative path')
        self.root_path = os.path.join(self.root_path, '')
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']
        self.dataset = config['dataset'].lower()

        if self.dataset == "weibo":
            self.train_path = self.root_path + 'train_origin.csv'
            self.val_path = self.root_path + 'val_origin.csv'
            self.test_path = self.root_path + 'test_origin.csv'
            self.category_dict = {
                "Economy": 0, "Health": 1, "Military": 2, "Science": 3,
                "Politics": 4, "International": 5, "Education": 6, "Entertainment": 7, "Society": 8
            }
        elif self.dataset == "weibo21":
            self.train_path = self.root_path + 'train_datasets.xlsx'
            self.val_path = self.root_path + 'val_datasets.xlsx'
            self.test_path = self.root_path + 'test_datasets.xlsx'
            self.category_dict = {
                "Technology": 0, "Military": 1, "Education": 2, "Disaster": 3,
                "Politics": 4, "Healthcare": 5, "Finance": 6, "Entertainment": 7, "Society": 8
            }
        elif self.dataset == "finefake":
            # Use the fixed paper-protocol classification partitions.
            self.train_path = self.root_path + config.get('finefake_train', 'train.csv')
            self.val_path = self.root_path + config.get('finefake_val', 'val.csv')
            self.test_path = self.root_path + config.get('finefake_test', 'test.csv')
            if self.val_path == self.test_path:
                raise ValueError("[ERROR] finefake val and test paths are identical - "
                                 "this leaks the test set into model selection.")
            self.category_dict = {
                "Politics": 0, "Entertainment": 1, "Business": 2,
                "Health": 3, "Society": 4, "Conflict": 5, "Uncategorized": 6
            }
        # Resolve aliases before checking that the three splits are distinct.
        paths = (self.train_path, self.val_path, self.test_path)
        for index, path in enumerate(paths):
            for previous in paths[:index]:
                if os.path.normcase(os.path.realpath(path)) == os.path.normcase(os.path.realpath(previous)) or os.path.exists(path) and os.path.exists(previous) and os.path.samefile(path, previous):
                    raise ValueError('Train, validation, and test files must be distinct')

    def get_dataloader(self, dataset):
        loader = None
        if self.emb_type == 'bert':
            if dataset == "weibo":
                from utils.clipld import bert_data as weibo_data
                loader = weibo_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                                    category_dict=self.category_dict, num_workers=self.num_workers)
            elif dataset == "weibo21":
                from utils.social import bert_data as weibo21_data
                loader = weibo21_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                                      category_dict=self.category_dict, num_workers=self.num_workers)
            elif dataset == "finefake":
                from utils.fld import bert_data as finefake_data
                loader = finefake_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                                       category_dict=self.category_dict, num_workers=self.num_workers)

        if dataset == "weibo":
            train_loader = loader.load_data(self.train_path, self.root_path + 'train_loader.pkl', self.root_path + 'train_clip_loader.pkl', True)
            val_loader = loader.load_data(self.val_path, self.root_path + 'val_loader.pkl', self.root_path + 'val_clip_loader.pkl', False)
            test_loader = loader.load_data(self.test_path, self.root_path + 'test_loader.pkl', self.root_path + 'test_clip_loader.pkl', False)
        elif dataset == "weibo21":
            train_loader = loader.load_data(self.train_path, self.root_path + 'train_loader.pkl', self.root_path + 'train_clip_loader.pkl', True)
            val_loader = loader.load_data(self.val_path, self.root_path + 'val_loader.pkl', self.root_path + 'val_clip_loader.pkl', False)
            test_loader = loader.load_data(self.test_path, self.root_path + 'test_loader.pkl', self.root_path + 'test_clip_loader.pkl', False)
        elif dataset == "finefake":
            ff_dir = self.root_path
            train_loader = loader.load_data(self.train_path, ff_dir + 'f_train_loader.pkl',
                                            ff_dir + 'f_train_clip.pkl', True)
            val_loader = loader.load_data(self.val_path, ff_dir + 'f_val_loader.pkl',
                                          ff_dir + 'f_val_clip.pkl', False)
            test_loader = loader.load_data(self.test_path, ff_dir + 'f_test_loader.pkl',
                                           ff_dir + 'f_test_clip.pkl', False)

        return train_loader, val_loader, test_loader

    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        from utils.partition import audit_files
        from utils.official import verify_files
        paths = {'train': self.train_path, 'val': self.val_path, 'test': self.test_path}
        audit_files(paths, self.root_path)
        verify_files(self.dataset, paths, self.configinfo.get('official_root') or self.root_path)
        train_loader, val_loader, test_loader = self.get_dataloader(self.dataset)

        if self.model_name == 'DDIN':
            from model.net import Trainer as DDINTrainer
            trainer = DDINTrainer(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
                                    use_cuda=self.use_cuda, lr=self.lr, train_loader=train_loader, dropout=self.dropout,
                                    weight_decay=self.weight_decay, val_loader=val_loader, test_loader=test_loader,
                                    category_dict=self.category_dict, early_stop=self.early_stop, epoches=self.epoch,
                                    save_param_dir=os.path.join(self.save_param_dir, self.model_name))
            return trainer.train()
        elif self.model_name == 'Gossip':
            from model.gossip import Trainer as GossipTrainer
            trainer = GossipTrainer(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims,
                                    bert_path_or_name=self.bert,
                                    clip_path_or_name='ViT-B-16',
                                    use_cuda=self.use_cuda, lr=self.lr, dropout=self.dropout,
                                    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                    category_dict=self.category_dict, weight_decay=self.weight_decay,
                                    save_param_dir=os.path.join(self.save_param_dir, self.model_name),
                                    early_stop=self.early_stop, epoches=self.epoch)
            return trainer.train()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
