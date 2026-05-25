import json
import os
import shutil
import re
import random
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "VisualNews", "origin")
JSON_NAME = "data.json"

NEW_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "ProcessedVisualNews")
ARTICLES_PATH = os.path.join(NEW_DATASET_PATH, "articles")
RELATIVE_ARTICLES_PATH = "./articles/"
IMAGES_PATH = os.path.join(NEW_DATASET_PATH, "images")
RELATIVE_IMAGES_PATH = './images/'
NUM_ARTICLES = 100_000

SEED = 1337

random.seed(SEED)

extension_re = re.compile(r"\.(\w+)$")

with open(os.path.join(DATASET_PATH, JSON_NAME), "r") as f:
    loaded_json = json.load(f)

dataset_length = len(loaded_json)

os.makedirs(NEW_DATASET_PATH, exist_ok=True)
os.makedirs(ARTICLES_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)

new_json = []

num_set = set()

for i in tqdm(range(NUM_ARTICLES)):
    # rand_num = random.randint(0, dataset_length-1)
    # while rand_num in num_set:
    #     rand_num = random.randint(0, dataset_length-1)
    
    # num_set.add(rand_num)
    article = loaded_json[i]
    extension_img = re.search(extension_re, article["image_path"])
    
    new_dict = {
        'id': i,
        'caption': article["caption"],
        'image_path': f"{RELATIVE_IMAGES_PATH}{i}.{extension_img.group(1)}",
        'article_path': f"{RELATIVE_ARTICLES_PATH}{i}.txt"
    }

    real_img_path = os.path.join(IMAGES_PATH, f"{i}.{extension_img.group(1)}")
    real_art_path =  os.path.join(ARTICLES_PATH, f"{i}.txt")

    origin_article = os.path.join(DATASET_PATH, article["article_path"])
    origin_image = os.path.join(DATASET_PATH, article["image_path"])
    try:
        shutil.copyfile(origin_article, real_art_path)
        shutil.copyfile(origin_image, real_img_path)
    except:
        print(article)
        raise ValueError

    new_json.append(new_dict)

with open(os.path.join(NEW_DATASET_PATH, "data.json"), "w") as g:
    json.dump(new_json, g)
