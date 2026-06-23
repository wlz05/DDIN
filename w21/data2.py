# -*-codeing = utf-8 -*-
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
def download_image(url, folder):
    response = requests.get(url,verify=False)
    if response.status_code == 200:
        filename = url.split("/")[-1]
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as file:
            file.write(response.content)
        print(f"Saved image: {filename}")
    else:
        print("Failed to download image, HTTP error:", response.status_code)

for i in range(2):
    json_file = json_files[i]
    f = open(json_file,encoding='utf-8')
    line = f.readline()  # id,content,comments,timestamp,piclists,label,category
    while line:
        try:
            folder = folders[i]
            os.makedirs(folder, exist_ok=True)
            images_set = set(os.listdir(folder))
            item = json.loads(line)
            record = {}
            piclists = item["piclists"]
            image_name = ""
            if isinstance(piclists,float): piclists = []
            if not isinstance(piclists,list): piclists = [piclists]
            for full_image_name in piclists:
                if full_image_name[:8]=='https://': full_image_name = "https://i0.wp.com/"+full_image_name[8:]
                if full_image_name[:7]=='http://': full_image_name = "https://i0.wp.com/"+full_image_name[7:]
                if full_image_name[:2]=='//': full_image_name = "https://i0.wp.com/"+full_image_name[2:]
                short_name = full_image_name[full_image_name.rfind('/') + 1:]
                if short_name[-4:]!='gif':
                    if short_name not in images_set:
                        if conduct_download:
                            print(full_image_name, folder + short_name)
                            download_image(full_image_name, folder)
                            image_name = image_name + "rumor_images/" + short_name + "|"
                            print("Download ok. {}".format(full_image_name))
                        else:
                            print("Do not download. {}".format(full_image_name))
                    else:
                        image_name = image_name + "rumor_images/" + short_name + "|"
                        print("Already Downloaded. {}".format(full_image_name))
                else:
                    print("Gif Skipped. {}".format(full_image_name))

            record['source'] = ""
            record['images'] = image_name
            record['content'] = item["content"]
            record['label'] = i
            results.append(record)
            line = f.readline()
        except Exception:
            print("Load JSON error!")
            error_json += 1
    f.close()

df = pd.DataFrame(results)
df.to_excel('real_datasets.xlsx')
