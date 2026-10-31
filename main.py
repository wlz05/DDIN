# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

import os
import argparse
import torch
import numpy as np
import json
from run import Run, SEEDS, run_seeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='DDIN', choices=['DDIN', 'Gossip'])
    # Normalize dataset names before selecting a loader.
    parser.add_argument('--dataset', default='weibo21', type=str.lower, choices=['weibo', 'weibo21', 'finefake'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--early_stop', type=int, default=5)  # tolerance of 5 to avoid early interruption
    parser.add_argument('--bert_vocab_file', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
    parser.add_argument('--root_path', default=None)
    parser.add_argument('--official_root', default=None)
    parser.add_argument('--bert', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch')
    parser.add_argument('--batchsize', type=int, default=64)
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument('--seed', type=int, help='Run one explicit training seed')
    seed_options.add_argument('--seeds', type=int, nargs='+', help='Run the declared seed grid; default: 42 52 3047 3407')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--bert_emb_dim', type=int, default=768)
    parser.add_argument('--w2v_emb_dim', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)    # tune LR here for new architectures
    parser.add_argument('--emb_type', default='bert', choices=['bert'])
    parser.add_argument('--w2v_vocab_file', default='./pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv')
    parser.add_argument('--save_param_dir', default= './param_model')

    args = parser.parse_args()
    if min(args.epoch, args.batchsize, args.max_len, args.early_stop) < 1 or args.num_workers < 0:
        parser.error('Epochs, batch size, sequence length, and patience must be positive; workers must be nonnegative')
    seeds = args.seeds if args.seeds is not None else [args.seed] if args.seed is not None else list(SEEDS)
    if len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2 ** 32 for seed in seeds):
        parser.error('Seeds must be distinct unsigned 32-bit integers')
    args.seed = seeds[0]
    if not np.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.seed < 2 ** 32:
        parser.error('Learning rate must be finite and positive; seed must be an unsigned 32-bit integer')
    if args.max_len < 2 or args.bert_emb_dim < 8 or args.bert_emb_dim % 8:
        parser.error('BERT needs at least two tokens and a hidden dimension divisible by eight')
    args.dataset = args.dataset.lower()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.emb_type == 'bert':
        emb_dim = args.bert_emb_dim
        vocab_file = args.bert_vocab_file
    elif args.emb_type == 'w2v':
        emb_dim = args.w2v_emb_dim
        vocab_file = args.w2v_vocab_file

    print(f'lr: {args.lr}; model name: {args.model_name} (DDIN Core); emb_type: {args.emb_type}; '
          f'batchsize: {args.batchsize}; epoch: {args.epoch}; gpu: {args.gpu}; emb_dim: {emb_dim}')

    config = {
        'use_cuda': torch.cuda.is_available(),
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'vocab_file': vocab_file,
        'emb_type': args.emb_type,
        'bert': args.bert,
        'root_path': args.root_path,
        'official_root': args.official_root,
        'weight_decay': 0.01,
        'model': {
            'mlp': {'dims': [384], 'dropout': 0.2}
        },
        'emb_dim': emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_param_dir': args.save_param_dir,
        'dataset': args.dataset
    }
    if args.dataset == 'finefake':
        # Keep the paper split fixed across training seeds.
        config['finefake_train'] = 'train.csv'
        config['finefake_val'] = 'val.csv'
        config['finefake_test'] = 'test.csv'
        print('FineFake 7-domain categories: Politics, Entertainment, Business, Health, Society, Conflict, Uncategorized')

    try:
        runner = Run(config=config)
    except ValueError as error:
        parser.error(str(error))
    try:
        result = run_seeds(config, seeds)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, 'Error: ' + str(error) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
