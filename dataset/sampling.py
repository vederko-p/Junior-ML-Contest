
import os
import shutil

import random
from math import ceil
from collections import defaultdict
from typing import Tuple, Callable

from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataset import indexing


def sample_data(
        ds_path: str,
        output_path: str,
        index_func: Callable[[tuple], tuple],
        out_folder_name: str,
        min_th: int = 1000,
        max_th: int = None,
        random_state: int = None,
) -> dict:
    """Random sample images from dataset by index.

    Parameters
    ----------
    ds_path : `str`
        Dataset path.
    output_path : `str`
        Path for sampled dataset output.
    index_func : `Callable`
        Function that define the folder index slice method to describes object
        index.
    out_folder_name : `str`
        Output folder name for sampled images.
    min_th, max_th : `int`, `int`
        Minimal and maximal thresholds values of images amount to sample.
    random_state : `int`
        Random state for sampler.

    Returns
    -------
    sampled_images : `dict`
        Dict os sampled images: {mark / mark+model: images}.
    """
    folders = os.listdir(ds_path)
    indexes = indexing.get_indexes(folders)
    collected_images = collect_images(ds_path, folders, indexes, index_func)
    sampled_images = sample_images(collected_images,
                                   min_th, max_th,
                                   random_state)
    user_answer = check_size(sampled_images)
    if user_answer:
        make_images_copy(sampled_images, output_path, out_folder_name)
    return sampled_images


def collect_images(
        ds_path: str,
        folders: list,
        indexes: list,
        index_func: Callable[[tuple], tuple]
) -> defaultdict:
    """Collect images by index."""
    imgs_by_marks = defaultdict(list)
    iterator = tqdm(zip(folders, indexes), total=len(folders))
    for fold_n, indx in iterator:
        imgs_loc_path = os.path.join(ds_path, fold_n)
        imgs_paths = os.listdir(imgs_loc_path)
        imgs_gl_paths = [os.path.join(imgs_loc_path, ilp) for ilp in imgs_paths]
        imgs_by_marks[index_func(indx)].extend(imgs_gl_paths)
    return imgs_by_marks


def sample_images(collected: dict,
                  min_th: int = 1000,
                  max_th: int = None,
                  random_state: int = None) -> defaultdict:
    """Sample images within each instance mark / mark+model)."""
    sampled_images = defaultdict(list)
    for mark, images in collected.items():
        images_len = len(images)
        if images_len > min_th:
            q = min(images_len, max_th) if max_th is not None else images_len
            handle_random_state(random_state)
            sampled_images[mark].extend(random.sample(images, q))
    return sampled_images


def handle_random_state(random_state: int = None) -> None:
    if random_state is not None:
        random.seed(random_state)


def check_size(imgs: dict) -> bool:
    """Check size of images."""
    total_bytes_size = sum(get_files_size(ips) for ips in imgs.values())
    s, tag = bytes_format(total_bytes_size)
    user_answer = user_call(s, tag)
    return user_answer


def get_files_size(paths: list) -> int:
    sb = sum([os.path.getsize(ip) for ip in paths])
    return sb


def bytes_format(sb: int) -> Tuple[float, str]:
    """Transform bytes into readable format."""
    tags = ['Bytes', 'KB', 'MB', 'GB']
    current_size = sb
    for t in tags:
        if int(current_size / 2**10) == 0 or t == 'GB':
            break
        current_size /= 2**10
    return ceil(current_size), t


def user_call(s, tag) -> bool:
    """Ask user to complete sampling due to needed free space."""
    text = f'{s} {tag} of free space are needed to complete sampling. ' \
           f'Sample? [y/n]: '
    while True:
        inp = input(text)
        if inp == 'y':
            return True
        elif inp == 'n':
            return False
        else:
            print('type either "y" or "n".')


def make_images_copy(imgs: dict, output_path: str, folder_name: str) -> None:
    """Copy folders and images into new dataset folder."""
    folders = collect_folders(imgs)
    output_folderpath = os.path.join(output_path, folder_name)
    create_folders(output_folderpath, folders)
    copy_images(output_folderpath, imgs)


def collect_folders(imgs: dict) -> set:
    """Collect folders from images paths."""
    folders = set()
    for imgs_lst in imgs.values():
        for ip in imgs_lst:
            fold = os.path.split(os.path.split(ip)[0])[-1]
            folders.add(fold)
    return folders


def create_folders(output_fp: str, folders: set) -> None:
    """Create output folder and subfolders for images."""
    os.mkdir(output_fp)
    for fold in folders:
        fold_p = os.path.join(output_fp, fold)
        os.mkdir(fold_p)


def copy_images(output_fp, imgs: dict) -> None:
    """Copy images from source dataset to output."""
    for img_lst in tqdm(imgs.values()):
        for ip in img_lst:
            img_folder = os.path.split(os.path.split(ip)[0])[-1]
            img_name = os.path.split(ip)[-1]
            img_out_path = os.path.join(output_fp, img_folder, img_name)
            shutil.copy(ip, img_out_path)


def imgs_distr_for_models(
    ds_path: str,
    xlim: Tuple[int, int] = None
) -> None:
    """Prepare data and plot images amount distribution for models."""
    distr_data = get_imgs_distr_m_data(ds_path)
    plot_distr_m(distr_data, xlim=xlim)


def get_imgs_distr_m_data(ds_path: str) -> list:
    """Prepare data to plot images amount distribution."""
    def indx_f(index):
        return index[0], index[1]

    folders = os.listdir(ds_path)
    indexes = [indx_f(indx) for indx in indexing.get_indexes(folders)]
    counter = defaultdict(int)
    iterator = tqdm(zip(folders, indexes), total=len(folders))
    for fold, indx in iterator:
        imgs_loc_path = os.path.join(ds_path, fold)
        imgs_len = len(os.listdir(imgs_loc_path))
        counter[indx] += imgs_len
    res = list(counter.values())
    return res


def plot_distr_m(distr_data: list, xlim: Tuple[int, int] = None):
    """Plot images amount distribution by models."""
    bins = range(max(distr_data) + 1)
    median = np.median(distr_data).round(2)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(distr_data, bins=bins, label=f'median: {median}')
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title('Распределение по числу изображений', fontsize=15)
    ax.set_xlabel('Число изображений', fontsize=13)
    ax.set_ylabel('Частота', fontsize=13)
    plt.legend(fontsize=13)
    plt.show()
