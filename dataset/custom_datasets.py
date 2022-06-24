
import os
from typing import Tuple, Callable, List
from collections import defaultdict
from random import sample, choice

import torch
from torch.utils.data import Dataset
import albumentations as A

from dataset import indexing
from dataset.utils import image_process as img_proc


def collect_dataset(
    ds_path: str,
    index_func: Callable[[tuple], tuple]
) -> Tuple[list, list, dict]:
    """Collect dataset images and labels."""
    images, labels, labels_code = [], [], {}
    folders = os.listdir(ds_path)
    targ_indexes = [index_func(i) for i in indexing.get_indexes(folders)]
    for fold, t_indx in zip(folders, targ_indexes):
        labels_code = update_labels_code(labels_code, t_indx)
        imgs_paths = get_imgs_paths(os.path.join(ds_path, fold))
        labels.extend([labels_code[t_indx]]*len(imgs_paths))
        images.extend(imgs_paths)
    return images, labels, labels_code


def update_labels_code(lbls_code: dict, lbl) -> dict:
    if lbl in lbls_code.keys():
        return lbls_code
    else:
        new_c = len(lbls_code.keys())
        lbls_code[lbl] = new_c
        return lbls_code


def get_imgs_paths(imgs_path: str):
    imgs = os.listdir(imgs_path)
    imgs_glob_paths = [os.path.join(imgs_path, img) for img in imgs]
    return imgs_glob_paths


def collect_blocks(
    images: List[str],
    labels: List[tuple]
) -> defaultdict:
    """Split images into blocks by labels."""
    blocks = defaultdict(list)
    for img, lbl in zip(images, labels):
        blocks[lbl].append(img)
    return blocks


class TLClassifyDataset(Dataset):
    def __init__(self,
                 ds_path: str,
                 index_func: Callable[[tuple], tuple],
                 crop: bool = False,
                 transform: A.Compose = None):
        self.crop = crop
        self.transform = transform

        ds = collect_dataset(ds_path, index_func)
        self.images, self.labels, self.labels_code = ds

        self.ds_len = len(self.labels)

    def __len__(self):
        return self.ds_len

    def __getitem__(self, indx):
        lbl_tens = torch.tensor(self.labels[indx])
        array_img = img_proc.open_image(self.images[indx])
        if self.crop:
            array_img = img_proc.custom_crop(array_img)
        if self.transform is not None:
            # augs : ([0, 1] np img) --> ([0, 1] np img)
            # np img dims ~ (h,w,c)
            array_img = self.transform(image=array_img)['image']
        img_tens = torch.permute(torch.FloatTensor(array_img), (2, 0, 1))
        return img_tens, lbl_tens
