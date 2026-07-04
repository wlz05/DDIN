# -*-codeing = utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

import pickle
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
import pandas as pd
import os
import numpy as np

def _init_fn(worker_id):
    np.random.seed(2024)

def read_pkl(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def word2input(texts, vocab_file, max_len):
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"[ERROR] BERT vocab file not found: {vocab_file}")
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    skipped = 0
    for i, text in enumerate(texts):
        if text is None or (isinstance(text, float) and pd.isna(text)) or not isinstance(text, str):
            skipped += 1
            text = ""
        text = text.strip()
        if len(text) == 0:
            text = " "
        try:
            encoded = tokenizer.encode(text, max_length=max_len, add_special_tokens=True,
                                       padding="max_length", truncation=True)
            token_ids.append(encoded)
        except Exception:
            skipped += 1
            token_ids.append([0] * max_len)
    if skipped > 0:
        print(f"[INFO] word2input: {skipped}/{len(texts)} texts repaired")
    token_ids = torch.tensor(token_ids)
    masks = (token_ids != 0).long()
    return token_ids, masks

class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self, path, imagepath, clipimagepath, shuffle, text_only=False):
        try:
            self.data = pd.read_csv(path, encoding="utf-8")
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to read CSV {path}: {e}")

        text_col = None
        for col in ["content", "text", "post_text"]:
            if col in self.data.columns:
                text_col = col
                break
        if text_col is None:
            raise KeyError(f"[ERROR] No text column found in {path}")

        label_col = "label"
        if label_col not in self.data.columns:
            raise KeyError(f"[ERROR] No label column in {path}")

        original_len = len(self.data)
        self.data = self.data.dropna(subset=[text_col])
        self.data[text_col] = self.data[text_col].fillna("").astype(str)
        if len(self.data) < original_len:
            print(f"[INFO] Dropped {original_len - len(self.data)} rows with missing content")

        if len(self.data) == 0:
            raise ValueError(f"[ERROR] No valid data after filtering in {path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clipmodel, _ = load_from_name("ViT-B-16", device=device, download_root="./")

        content = self.data[text_col].to_numpy()
        label = torch.tensor(self.data[label_col].astype(int).to_numpy())

        if "category" in self.data.columns and self.category_dict:
            category = torch.tensor(self.data["category"].apply(
                lambda c: self.category_dict.get(str(c), 0)).to_numpy())
        elif "domain" in self.data.columns and self.category_dict:
            category = torch.tensor(self.data["domain"].apply(
                lambda c: self.category_dict.get(str(c), 0)).to_numpy())
        else:
            category = torch.zeros(len(self.data), dtype=torch.long)

        token_ids, masks = word2input(content, self.vocab_file, self.max_len)

        try:
            ordered_image = pickle.load(open(imagepath, "rb"))
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to load image pickle {imagepath}: {e}")
        try:
            clip_image = pickle.load(open(clipimagepath, "rb"))
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to load CLIP image pickle {clipimagepath}: {e}")

        try:
            clip_text = clip.tokenize(list(content))
        except Exception as e:
            print(f"[WARNING] CLIP tokenize batch failed: {e}, tokenizing individually...")
            clip_tokens = []
            for t in content:
                try:
                    ct = clip.tokenize([str(t) if t else " "])
                    clip_tokens.append(ct[0])
                except Exception:
                    clip_tokens.append(torch.zeros(77, dtype=torch.long))
            clip_text = torch.stack(clip_tokens)

        n_samples = len(token_ids)
        assert ordered_image.size(0) == n_samples, \
            f"Image pickle row count mismatch: {ordered_image.size(0)} vs {n_samples}"
        assert clip_image.size(0) == n_samples, \
            f"CLIP image pickle row count mismatch: {clip_image.size(0)} vs {n_samples}"
        assert clip_text.size(0) == n_samples, \
            f"CLIP text token count mismatch: {clip_text.size(0)} vs {n_samples}"

        datasets = TensorDataset(token_ids, masks, label, category, ordered_image, clip_image, clip_text)
        dataloader = DataLoader(dataset=datasets, batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=True, shuffle=shuffle, worker_init_fn=_init_fn)
        print(f"[INFO] FineFake Dataloader: {n_samples} samples, batch_size={self.batch_size}")
        return dataloader

# Author: 
