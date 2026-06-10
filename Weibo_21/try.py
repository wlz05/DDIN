import json
import wget
import os
import pandas as pd
import requests
results, record, error_json = [], {}, 0
json_files = ['fake_release_all.json','real_release_all.json']
folders = ['rumor_images','nonrumor_images']
conduct_download = True
folder = folders[0]
images_set = set(os.listdir(folder))
print(len(images_set))