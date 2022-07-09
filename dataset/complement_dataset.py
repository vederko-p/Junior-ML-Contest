
import os
from random import sample
from typing import Tuple

from dataset import indexing
from dataset.utils.handle_random_state import handle_random_state


def complement_dataset(
    ds_path: str, target_marks: set,
    n: int, random_state: int = None
) -> Tuple[list, list]:
    base_folders = os.listdir(ds_path)
    base_indexes = indexing.get_indexes(base_folders)
    new_images = []
    new_indexes = []
    for fold, indx in zip(base_folders, base_indexes):
        if indx[0] in target_marks:
            imgs_path = os.path.join(ds_path, fold)
            imgs_loc_paths = os.listdir(imgs_path)
            imgs_glob_paths = [os.path.join(imgs_path, ip) for ip in imgs_loc_paths]
            new_images.extend(imgs_glob_paths)
            new_indexes.extend([indx]*len(imgs_glob_paths))
    handle_random_state(random_state)
    imgs_indexes = sample(range(len(new_images)), n)
    out_new_images = [new_images[t_indx] for t_indx in imgs_indexes]
    out_new_indexes = [new_indexes[t_indx] for t_indx in imgs_indexes]
    return out_new_images, out_new_indexes
