# -*-codeing = utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

import pickle
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
import pandas as pd
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image


def read_image():
    """Load and preprocess images, use black placeholder for corrupted ones."""
    image_list = {}
    file_list = ['data/nonrumor_images/', 'data/rumor_images/']
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for path in file_list:
        if not os.path.exists(path):
            print(f"[WARNING] Image directory not found: {path}, skipping...")
            continue

        for i, filename in enumerate(os.listdir(path)):
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except Exception as e:
                print(f"[WARNING] Corrupted image: {path}{filename}, using black placeholder. Error: {e}")
                # Black placeholder
                placeholder = torch.zeros(3, 224, 224)
                image_list[filename.split('/')[-1].split(".")[0].lower()] = placeholder

    print(f"[INFO] Loaded {len(image_list)} images total")
    return image_list


def _init_fn(worker_id):
    np.random.seed(2024)


def read_pkl(path):
    """Safely load a pickle file."""
    try:
        with open(path, "rb") as f:
            t = pickle.load(f)
        return t
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] Pickle file not found: {path}")
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"[ERROR] Corrupted pickle file: {path}, error: {e}")


def df_filter(df_data):
    """Filter data with undetermined category."""
    df_data = df_data[df_data['category'] != 'cannot determine']
    return df_data


def word2input(texts, vocab_file, max_len):
    """
    BERT tokenization with automatic fallback for missing/invalid text.
    """
    # Check vocab file
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"[ERROR] BERT vocab file not found: {vocab_file}")

    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    skipped_count = 0

    for i, text in enumerate(texts):
        # Handle None, NaN, non-string
        if text is None or (isinstance(text, float) and pd.isna(text)) or not isinstance(text, str):
            skipped_count += 1
            if skipped_count <= 5:
                print(f"[WARNING] Text at index {i} is missing or invalid type ({type(text).__name__}), using empty string")
            text = ""

        # Trim whitespace, use single space for empty strings to avoid tokenizer errors
        text = text.strip()
        if len(text) == 0:
            text = " "

        try:
            encoded = tokenizer.encode(
                text,
                max_length=max_len,
                add_special_tokens=True,
                padding='max_length',
                truncation=True
            )
            token_ids.append(encoded)
        except Exception as e:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"[WARNING] Tokenizer failed for text at index {i}: '{str(text)[:50]}...', error: {e}")
            # Zero fallback
            token_ids.append([0] * max_len)

    if skipped_count > 0:
        print(f"[INFO] word2input: {skipped_count}/{len(texts)} texts were repaired with fallback")

    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i, token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids, masks


class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self, path, imagepath, clipimagepath, shuffle, text_only=False):
        # Read CSV data file
        try:
            self.data = pd.read_csv(path, encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"[ERROR] Data CSV not found: {path}")
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to read CSV {path}: {e}")

        # Validate required columns
        required_cols = ['content', 'label', 'category']
        missing_cols = [c for c in required_cols if c not in self.data.columns]
        if missing_cols:
            raise KeyError(f"[ERROR] Missing required columns in {path}: {missing_cols}")

        # Remove rows with missing text
        original_len = len(self.data)
        self.data = self.data.dropna(subset=['content'])
        self.data['content'] = self.data['content'].fillna('').astype(str)
        if len(self.data) < original_len:
            print(f"[INFO] Dropped {original_len - len(self.data)} rows with missing content in {path}")

        if len(self.data) == 0:
            raise ValueError(f"[ERROR] No valid data after filtering in {path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clipmodel, _ = load_from_name("ViT-B-16", device=device, download_root='./')

        # Extract text, labels, categories
        content = self.data['content'].to_numpy()
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())

        # BERT tokenization (word2input handles errors)
        token_ids, masks = word2input(content, self.vocab_file, self.max_len)

        # Safely load pre-computed image feature pickle
        try:
            ordered_image = pickle.load(open(imagepath, 'rb'))
        except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"[ERROR] Failed to load image pickle {imagepath}: {e}")

        try:
            clip_image = pickle.load(open(clipimagepath, 'rb'))
        except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"[ERROR] Failed to load CLIP image pickle {clipimagepath}: {e}")

        # CLIP text tokenization (with fallback)
        try:
            clip_text = clip.tokenize(list(content))
        except Exception as e:
            print(f"[WARNING] CLIP tokenize batch failed: {e}, tokenizing individually...")
            clip_tokens = []
            for i, t in enumerate(content):
                try:
                    ct = clip.tokenize([str(t) if t else " "])
                    clip_tokens.append(ct[0])
                except Exception:
                    clip_tokens.append(torch.zeros(77, dtype=torch.long))  # CLIP max context length
            clip_text = torch.stack(clip_tokens)

        # Ensure consistent row count across all tensors
        n_samples = len(token_ids)
        assert ordered_image.size(0) == n_samples, \
            f"Image pickle row count mismatch: {ordered_image.size(0)} vs {n_samples}"
        assert clip_image.size(0) == n_samples, \
            f"CLIP image pickle row count mismatch: {clip_image.size(0)} vs {n_samples}"
        assert clip_text.size(0) == n_samples, \
            f"CLIP text token count mismatch: {clip_text.size(0)} vs {n_samples}"

        datasets = TensorDataset(
            token_ids, masks, label, category,
            ordered_image, clip_image, clip_text
        )
        dataloader = DataLoader(
            dataset=datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        print(f"[INFO] Dataloader created: {n_samples} samples, batch_size={self.batch_size}")
        return dataloader
