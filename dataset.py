# -*-codeing = utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

import random
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import logging
from transformers import BertTokenizer, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)

class dataset(data.Dataset):
    def __init__(
        self,
        root_path,
        bert_tokenizer_instance,
        clip_processor_instance,
        dataset_name="gossip",
        image_size=224,
        is_train=True,
        data_augment=False,
        duplicate_fake_times=0,
        bert_max_len=197,
        clip_max_len=77,
        category_dict=None
    ):
        self.bert_tokenizer = bert_tokenizer_instance
        self.clip_processor = clip_processor_instance
        if self.bert_tokenizer is None:
             logger.warning("BERT tokenizer instance is None.")
        if self.clip_processor is None:
             logger.warning("CLIP processor instance is None.")

        self.duplicate_fake_times = duplicate_fake_times
        self.dataset_name = dataset_name.lower()
        assert (
            self.dataset_name == "politi" or self.dataset_name == "gossip"
        ), "Error! Only 'gossip' or 'politi' are supported!"
        super(dataset, self).__init__()

        logger.info(f"dataset init: dataset_name='{self.dataset_name}', is_train={is_train}")
        logger.info(f"Duplicate fake (label=0) samples: {self.duplicate_fake_times}")

        self.is_train = is_train
        self.root_path = root_path
        self.image_size = image_size
        self.bert_max_len = bert_max_len
        self.clip_max_len = clip_max_len

        self.label_dict = []

        dataset_folder_path = self.root_path
        if not os.path.isdir(dataset_folder_path):
             logger.error(f"Dataset root is not a valid directory: {self.root_path}")
             raise FileNotFoundError(f"Dataset root invalid: {self.root_path}")

        csv_file_name = f"{self.dataset_name}_{'train' if self.is_train else 'test'}.csv"
        csv_path = os.path.join(dataset_folder_path, csv_file_name)
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found in {dataset_folder_path}: {csv_file_name}")
            raise FileNotFoundError(f"CSV file not found in {dataset_folder_path}: {csv_file_name}")

        logger.info(f"Loading data from CSV: {csv_path}")
        try:
            data_df = pd.read_csv(csv_path)
            potential_text_cols = ['content', 'text', 'post_text']
            fill_dict = {col: '' for col in potential_text_cols if col in data_df.columns}
            data_df.fillna(fill_dict, inplace=True)
        except Exception as e:
            logger.error(f"Failed to read CSV {csv_path}: {e}")
            raise

        content_col = None
        if 'post_text' in data_df.columns: content_col = 'post_text'
        elif 'content' in data_df.columns: content_col = 'content'
        elif 'text' in data_df.columns: content_col = 'text'
        else: raise ValueError(f"CSV {csv_path} missing text column.")
        logger.info(f"Using text column: '{content_col}'")

        image_id_col = None
        if 'image_id' in data_df.columns: image_id_col = 'image_id'
        elif 'image' in data_df.columns: image_id_col = 'image'
        else: raise ValueError(f"CSV {csv_path} missing image ID column.")
        logger.info(f"Using image ID column: '{image_id_col}'")

        label_col = 'label'
        if label_col not in data_df.columns: raise ValueError(f"CSV {csv_path} missing label column.")

        fake_news_num = 0  # label 0
        real_news_num = 0  # label 1
        valid_labels_count = 0
        if label_col in data_df.columns:
             for label_val in data_df[label_col]:
                 try:
                     label_int = int(label_val)
                     if label_int == 0: fake_news_num += 1
                     elif label_int == 1: real_news_num += 1
                     else: continue
                     valid_labels_count += 1
                 except (ValueError, TypeError): pass
        if valid_labels_count == 0: raise ValueError("Dataset has no valid labels.")

        if real_news_num == 0: self.pos_weight = torch.tensor(1.0); logger.warning("No real news samples (label=1). pos_weight set to 1.0.")
        else: self.pos_weight = torch.tensor(fake_news_num / real_news_num)
        self.thresh = real_news_num / valid_labels_count if valid_labels_count > 0 else 0.5
        logger.info(f"Valid labels: {valid_labels_count}, Fake (label=0): {fake_news_num}, Real (label=1): {real_news_num}")
        logger.info(f"Computed pos_weight (fake/real): {self.pos_weight:.4f}, thresh (real ratio): {self.thresh:.4f}")

        skipped_num = 0
        image_folder_name = f"{self.dataset_name}_{'train' if self.is_train else 'test'}"
        image_base_dir = os.path.join(dataset_folder_path, image_folder_name)
        if not os.path.isdir(image_base_dir): raise FileNotFoundError(f"Expected image directory not found: {image_base_dir}")
        logger.info(f"Image base directory: {image_base_dir}")

        category_col = None
        if 'category' in data_df.columns:
            category_col = 'category'
        elif 'domain' in data_df.columns:
            category_col = 'domain'

        for idx, row in data_df.iterrows():
            try: label = int(row[label_col]); assert label in [0, 1]
            except: skipped_num += 1; continue
            image_id = str(row[image_id_col]); content = str(row[content_col])
            potential_image_paths = [ os.path.join(image_base_dir, f"{image_id}{ext}") for ext in ['', '.jpg', '.png', '.jpeg']]
            full_image_path = next((p for p in potential_image_paths if os.path.exists(p) and os.path.isfile(p)), None)
            if full_image_path is None: skipped_num += 1; continue
            if category_col and category_dict:
                cat_str = str(row[category_col])
                category_val = category_dict.get(cat_str, 0)
            else:
                category_val = 0
            record = {"image_path": full_image_path, "label": label, "content": content, "category": category_val}
            self.label_dict.append(record)
            if record["label"] == 0 and self.is_train and self.duplicate_fake_times > 0:  # duplicate Fake (label=0)
                for _ in range(self.duplicate_fake_times): self.label_dict.append(record)
        if skipped_num > 0: logger.warning(f"Skipped {skipped_num} samples due to missing images or invalid labels")
        logger.info(f"Final dataset size (including duplicates): {len(self.label_dict)}")
        if not self.label_dict: raise ValueError("Dataset is empty.")

        self.mae_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        record = self.label_dict[index]
        image_path = record["image_path"]
        content = record["content"]
        label = record["label"]
        category = record["category"]

        try: img_pil = Image.open(image_path).convert('RGB')
        except Exception as e: logger.warning(f"Failed to load image {image_path}: {e}, using black placeholder."); img_pil = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        try: img_mae = self.mae_transform(img_pil)
        except Exception as e: logger.warning(f"MAE transform failed for {image_path}: {e}, using zeros"); img_mae = torch.zeros((3, self.image_size, self.image_size))

        clip_pixel_values = None; clip_default_size = 224
        if self.clip_processor and hasattr(self.clip_processor, 'image_processor'):
            try:
                clip_h = self.clip_processor.image_processor.size.get('height', clip_default_size)
                clip_w = self.clip_processor.image_processor.size.get('width', clip_default_size)
                clip_processed = self.clip_processor(images=img_pil, return_tensors="pt")
                clip_pixel_values = clip_processed['pixel_values'].squeeze(0)
            except Exception as e: logger.warning(f"CLIP image processing failed for {image_path}: {e}, using zeros"); clip_h = self.clip_processor.image_processor.size.get('height', clip_default_size); clip_w = self.clip_processor.image_processor.size.get('width', clip_default_size); clip_pixel_values = torch.zeros((3, clip_h, clip_w))
        else: clip_pixel_values = torch.zeros((3, clip_default_size, clip_default_size))

        bert_input_ids, bert_attention_mask = None, None
        if self.bert_tokenizer:
            try:
                bert_tokenized = self.bert_tokenizer(content, padding='max_length', truncation=True, max_length=self.bert_max_len, return_tensors="pt")
                bert_input_ids = bert_tokenized['input_ids'].squeeze(0)
                bert_attention_mask = bert_tokenized['attention_mask'].squeeze(0)
            except Exception as e: logger.warning(f"BERT text processing failed: {e}, using zeros")
        if bert_input_ids is None: bert_input_ids = torch.zeros(self.bert_max_len, dtype=torch.long); bert_attention_mask = torch.zeros(self.bert_max_len, dtype=torch.long)

        clip_input_ids, clip_attention_mask = None, None
        if self.clip_processor:
            try:
                clip_text_processed = self.clip_processor(text=content, return_tensors="pt", padding='max_length', truncation=True, max_length=self.clip_max_len)
                clip_input_ids = clip_text_processed['input_ids'].squeeze(0)
                clip_attention_mask = clip_text_processed['attention_mask'].squeeze(0)
            except Exception as e: logger.warning(f"CLIP text processing failed: {e}, using zeros")
        if clip_input_ids is None: clip_input_ids = torch.zeros(self.clip_max_len, dtype=torch.long); clip_attention_mask = torch.zeros(self.clip_max_len, dtype=torch.long)

        item = {'content': bert_input_ids,'content_masks': bert_attention_mask,'image': img_mae,'clip_image': clip_pixel_values,'clip_text': clip_input_ids,'clip_attention_mask': clip_attention_mask,'label': torch.tensor(label, dtype=torch.float),'category': torch.tensor(category, dtype=torch.long)}
        return item

    def __len__(self):
        return len(self.label_dict)

# Author: 
