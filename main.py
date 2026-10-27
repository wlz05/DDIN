# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

import os
import argparse
import torch
import numpy as np
import random
from run import Run

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
    parser.add_argument('--seed', type=int, default=42)
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
    if not np.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.seed < 2 ** 32:
        parser.error('Learning rate must be finite and positive; seed must be an unsigned 32-bit integer')
    args.dataset = args.dataset.lower()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    seed = args.seed
    # Seed every random generator before creating the model.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

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

    Run(config=config).main()

if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
