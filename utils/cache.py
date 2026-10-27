# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

from pathlib import Path

def cache_directory(name):
    # Keep downloaded weights beside the project for relocation.
    directories = {'clip': Path('model_weights/clip_cn'), 'huggingface': Path('model_weights/huggingface'), 'torch': Path('model_weights/checkpoints')}
    if name not in directories:
        raise ValueError('Unknown model cache')
    return './' + directories[name].as_posix()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
