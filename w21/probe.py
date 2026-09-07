# -*- coding: utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

import json
import wget
import os
import pandas as pd
import requests
results, record, error_json = [], {}, 0
json_files = ['fake_release_all.json','real_release_all.json']
folders = ['rumor_images','nonrumor_images']
conduct_download = True
if __name__ == '__main__':
    folder = folders[0]
    images_set = set(os.listdir(folder))
    print(len(images_set))

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
