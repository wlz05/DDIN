# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection (accepted by WISE 2026)

from contextlib import contextmanager
import math
import ntpath
import os
from pathlib import Path
import tempfile
import torch

class FGM:
    def __init__(self, model, epsilon=0.5):
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError('FGM epsilon must be finite and nonnegative')
        self.model, self.epsilon, self.backup = model, epsilon, {}

    @torch.no_grad()
    def attack(self, emb_name='bert.embeddings.word_embeddings.weight'):
        if self.backup:
            raise RuntimeError('Restore the previous perturbation before attacking again')
        updates = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad or emb_name not in name or param.grad is None:
                continue
            norm = torch.norm(param.grad.float())
            if not torch.isfinite(norm):
                raise FloatingPointError('FGM gradient is nonfinite')
            if norm > 0:
                updates.append((name, param, param.grad / norm))
        self.backup = {name: param.detach().clone() for name, param, _ in updates}
        try:
            for name, param, direction in updates:
                param.add_(direction, alpha=self.epsilon)
                if not torch.isfinite(param).all():
                    raise FloatingPointError('FGM perturbation is nonfinite')
        except BaseException:
            self.restore()
            raise

    @torch.no_grad()
    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        parameters = dict(self.model.named_parameters())
        for name, value in self.backup.items():
            parameters[name].copy_(value)
        self.backup.clear()

    @contextmanager
    def perturb(self):
        self.attack()
        try:
            yield
        finally:
            # Restore embeddings even when adversarial inference fails.
            self.restore()

class EMA:
    def __init__(self, model, decay=0.999):
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError('EMA decay must lie in [0, 1)')
        self.model, self.decay, self.shadow, self.backup = model, decay, {}, {}
        self.register()

    def register(self):
        if self.backup:
            raise RuntimeError('Cannot register EMA while averaged weights are active')
        self.shadow = {name: param.detach().clone() for name, param in self.model.named_parameters() if param.requires_grad}

    @torch.no_grad()
    def update(self):
        if self.backup:
            raise RuntimeError('Restore live weights before updating EMA')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].lerp_(param.detach(), 1 - self.decay)

    @torch.no_grad()
    def apply_shadow(self):
        if self.backup:
            raise RuntimeError('EMA weights are already active')
        parameters = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        if any(name not in self.shadow or self.shadow[name].shape != param.shape for name, param in parameters):
            raise RuntimeError('EMA state differs from the live model')
        self.backup = {name: param.detach().clone() for name, param in parameters}
        try:
            for name, param in parameters:
                param.copy_(self.shadow[name])
        except BaseException:
            self.restore()
            raise

    @torch.no_grad()
    def restore(self):
        parameters = dict(self.model.named_parameters())
        for name, value in self.backup.items():
            parameters[name].copy_(value)
        self.backup.clear()

    @contextmanager
    def average_parameters(self):
        self.apply_shadow()
        try:
            yield
        finally:
            # Keep live weights intact after validation or save errors.
            self.restore()

class WarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        if type(max_epochs) is not int or max_epochs < 1 or type(warmup_epochs) is not int or not 0 <= warmup_epochs <= max_epochs:
            raise ValueError('Scheduler epochs must satisfy 0 <= warmup <= maximum')
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        if not self.base_lrs or any(not math.isfinite(value) or value <= 0 for value in self.base_lrs):
            raise ValueError('Base learning rates must be finite and positive')
        if isinstance(eta_min, bool) or not isinstance(eta_min, (int, float)) or not math.isfinite(eta_min) or not 0 <= eta_min <= min(self.base_lrs):
            raise ValueError('Minimum learning rate must lie between zero and every base rate')
        self.optimizer, self.warmup_epochs, self.max_epochs = optimizer, warmup_epochs, max_epochs
        self.eta_min, self.current_epoch = eta_min, 0

    def step(self):
        # Hold the final schedule value after the last epoch.
        self.current_epoch = min(self.current_epoch + 1, self.max_epochs)
        for base, group in zip(self.base_lrs, self.optimizer.param_groups):
            if self.current_epoch <= self.warmup_epochs:
                value = base * self.current_epoch / self.warmup_epochs
            else:
                progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                value = self.eta_min + (base - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
            group['lr'] = value

def save_state(model, path):
    target = Path(path)
    if not str(path).strip() or target.is_absolute() or str(path).startswith(("/", "\\")) or ntpath.isabs(str(path)) or ntpath.splitdrive(str(path))[0]:
        raise ValueError('Checkpoint path must be a nonempty relative path')
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError('Cannot save nonfinite model weights')
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.checkpoint-', dir=target.parent)
    os.close(handle)
    try:
        with open(name, 'wb') as stream:
            torch.save(state, stream)
        # Replace the checkpoint only after serialization succeeds.
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
