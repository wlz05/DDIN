# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE2026)

import  torch

from model.layers import clip_fuion

a = torch.rand(64,1024)
clip_fusion = clip_fuion(1024, 320, [348], 0.1)
clip_fusion(a)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
