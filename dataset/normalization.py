
import os
import numpy as np
from PIL import Image
from typing import Tuple
from tqdm.notebook import tqdm


def get_normalize_params(ds_path: str) -> Tuple[float, float]:
    """Returns params for images normalization from collected images."""
    images_paths = collect_images(ds_path)
    params = evaluate_params(images_paths)
    return params


def collect_images(ds_path: str) -> list:
    """Returns paths of all dataset images."""
    print('\nCollecting images')
    folders = os.listdir(ds_path)
    imgs_paths_total = []
    for fold_n in tqdm(folders):
        imgs_loc_path = os.path.join(ds_path, fold_n)
        imgs = os.listdir(imgs_loc_path)
        imgs_paths = [os.path.join(imgs_loc_path, ilp) for ilp in imgs]
        imgs_paths_total.extend(imgs_paths)
    return imgs_paths_total


def evaluate_params(images_paths: list) -> Tuple[float, float]:
    """Returns params for images normalization."""
    print('\nEvaluating params')
    mean_wh_list = []
    std_wh_list = []
    for ip in tqdm(images_paths):
        img_array = np.asarray(Image.open(ip)) / 255
        mean_wh_list.append(img_array.mean())
        std_wh_list.append(img_array.std())
    mean = np.array(mean_wh_list).mean()
    std = np.array(std_wh_list).mean()
    return mean, std
