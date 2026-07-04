# -*-codeing = utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import os
import numpy as np
import torch
import pickle
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
def read_image():
    image_list = {}
    file_list = ['weibo/nonrumor_images/', 'weibo/rumor_images/']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    for path in file_list:
        if not os.path.exists(path):
            print(f"[WARNING] Image directory not found: {path}, skipping...")
            continue
        for i, filename in enumerate(os.listdir(path)):
            try:
                im = Image.open(path + filename)
                im = preprocess(im).unsqueeze(0).to(device)
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except Exception as e:
                print(f"[WARNING] Corrupted image: {path}{filename}, using zero placeholder. Error: {e}")
                placeholder = preprocess(Image.new('RGB', (224, 224), (0, 0, 0))).unsqueeze(0).to(device)
                image_list[filename.split('/')[-1].split(".")[0].lower()] = placeholder
    print(f"[INFO] Loaded {len(image_list)} images total")
    return image_list

def _init_fn(worker_id):
    np.random.seed(2021)

def read_pkl(path):
    with open(path,"rb")as f:
        t = pickle.load(f)
    return t
def df_filter(df_data):
    df_data = df_data[df_data['category'] != 'cannot determine']
    return df_data

class bert_data():
    def __init__(self,max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data_train(self,path,shuffle,text_only = False):
        self.data = pd.read_csv(path,encoding='utf-8')
        post = self.data
        ordered_image = []
        post_id = []
        image_id_list = []
        image_id = ""
        image = read_image()
        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image:
                    break

            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                post_id.append(id)

        ordered_image = torch.tensor([item.cpu().detach().numpy() for item in ordered_image]).squeeze(1)
        with open('data/train_clip_loader.pkl', 'wb') as file:
            pickle.dump(ordered_image, file)
        return 1
    def load_data_test(self,path,shuffle,text_only = False):
        self.data = pd.read_csv(path,encoding='utf-8')
        post = self.data
        ordered_image = []
        post_id = []
        image_id_list = []
        image_id = ""
        image = read_image()
        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image:
                    break

            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                post_id.append(id)

        ordered_image = torch.tensor([item.cpu().detach().numpy() for item in ordered_image]).squeeze(1)
        with open('data/test_clip_loader.pkl', 'wb') as file:
            pickle.dump(ordered_image, file)
        return 1
    def load_data_val(self,path,shuffle,text_only = False):
        self.data = pd.read_csv(path,encoding='utf-8')
        post = self.data
        ordered_image = []
        post_id = []
        image_id_list = []
        image_id = ""
        image = read_image()
        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image:
                    break

            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                post_id.append(id)

        ordered_image = torch.tensor([item.cpu().detach().numpy() for item in ordered_image]).squeeze(1)
        with open('data/val_clip_loader.pkl', 'wb') as file:
            pickle.dump(ordered_image, file)
        return 1
category_dict = {
        "Economy": 0,
        "Health": 1,
        "Military": 2,
        "Science": 3,
        "Politics": 4,
        "International": 5,
        "Education": 6,
        "Entertainment": 7,
        "Society": 8
}
loader = bert_data(max_len=170, batch_size=64, vocab_file='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
                   category_dict=category_dict, num_workers=1)

val_loader = loader.load_data_val("data/val_origin.csv", True)
test_loader = loader.load_data_test("data/test_origin.csv", True)
train_loader = loader.load_data_train("data/train_origin.csv", True)

# Author: 
# Corresponding Mail: 
