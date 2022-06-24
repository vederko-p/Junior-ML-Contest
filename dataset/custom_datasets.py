
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


class TripletLossDataset:
    def __init__(self,
                 ds_path: str,
                 index_func: Callable[[tuple], tuple],
                 length: int,
                 crop: bool = False,
                 transform: A.Compose = None):
        # length - количество триплетов в одной эпохе
        self.ds_len = length
        self.crop = crop
        self.transform = transform

        ds = collect_dataset(ds_path, index_func)
        images, labels, self.labels_code = ds
        self.blocks = collect_blocks(images, labels)
        self.blocks_range = range(len(self.blocks.keys()))

    def __len__(self):
        return self.ds_len

    def __getitem__(self, indx, img_meta: bool = False):
        ap_indx, n_indx = sample(self.blocks_range, 2)
        a_ip, p_ip = sample(self.blocks[ap_indx], 2)
        n_ip = choice(self.blocks[n_indx])
        triplet_sample = self.get_tl_sample(a_ip, p_ip, n_ip, img_meta)
        return triplet_sample

    def get_tl_sample(
            self, a_ip: str, p_ip: str, n_ip: str, img_meta: bool
    ) -> dict:
        a_img_tens, p_img_tens, n_img_tens = self.proc_imgs(a_ip, p_ip, n_ip)
        if img_meta:
            anc_name, pos_name, neg_name = self.get_meta(a_ip, p_ip, n_ip)
            sample = {'Anc': a_img_tens, 'Anc_n': anc_name,
                      'Pos': p_img_tens, 'Pos_n': pos_name,
                      'Neg': n_img_tens, 'Neg_n': neg_name}
        else:
            sample = {'Anc': a_img_tens,
                      'Pos': p_img_tens,
                      'Neg': n_img_tens}
        return sample

    def proc_imgs(
            self, a_ip: str, p_ip: str, n_ip: str
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        a_array_img = img_proc.open_image(a_ip)
        p_array_img = img_proc.open_image(p_ip)
        n_array_img = img_proc.open_image(n_ip)
        if self.crop:
            a_array_img = img_proc.custom_crop(a_array_img)
            p_array_img = img_proc.custom_crop(p_array_img)
            n_array_img = img_proc.custom_crop(n_array_img)
        if self.transform is not None:
            # augs : ([0, 1] np img) --> ([0, 1] np img)
            # np img dims ~ (h,w,c)
            a_array_img = self.transform(image=a_array_img)['image']
            p_array_img = self.transform(image=p_array_img)['image']
            n_array_img = self.transform(image=n_array_img)['image']
        a_img_tens = torch.permute(torch.FloatTensor(a_array_img), (2, 0, 1))
        p_img_tens = torch.permute(torch.FloatTensor(p_array_img), (2, 0, 1))
        n_img_tens = torch.permute(torch.FloatTensor(n_array_img), (2, 0, 1))
        return a_img_tens, p_img_tens, n_img_tens

    @staticmethod
    def get_meta(a_ip: str, p_ip: str, n_ip: str) -> Tuple[str, str, str]:
        csn = [os.path.split(os.path.split(a_ip)[0])[-1].split(' '),
               os.path.split(os.path.split(p_ip)[0])[-1].split(' '),
               os.path.split(os.path.split(n_ip)[0])[-1].split(' ')]
        cars_names = tuple([f'{sn[0]}\n{sn[1]}\n{sn[-1][:4]}' for sn in csn])
        return cars_names
