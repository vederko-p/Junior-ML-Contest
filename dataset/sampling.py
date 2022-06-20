
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

import dataset.utils.handle_random_state as hrs


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
    imgs_by_object = defaultdict(list)
    iterator = tqdm(zip(folders, indexes), total=len(folders))
    for fold_n, indx in iterator:
        imgs_loc_path = os.path.join(ds_path, fold_n)
        imgs_paths = os.listdir(imgs_loc_path)
        imgs_gl_paths = [os.path.join(imgs_loc_path, ilp) for ilp in imgs_paths]
        imgs_by_object[index_func(indx)].extend(imgs_gl_paths)
    return imgs_by_object


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
            hrs.handle_random_state(random_state)
            sampled_images[mark].extend(random.sample(images, q))
    return sampled_images


def check_size(imgs: dict) -> bool:
    """Check size of images."""
    total_bytes_size = sum(get_files_size(ips) for ips in imgs.values())
    s, tag = bytes_format(total_bytes_size)
    user_answer = user_call_to_copy(s, tag)
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


def user_call_to_copy(s, tag) -> bool:
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


def make_images_copy(imgs: dict, output_path: str,
                     folder_name: str, del_from_src: bool = False) -> None:
    """Copy folders and images into new dataset folder."""
    folders = collect_folders(imgs)
    output_folderpath = os.path.join(output_path, folder_name)
    create_folders(output_folderpath, folders)
    copy_images(output_folderpath, imgs, del_from_src)


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


def copy_images(output_fp, imgs: dict, del_from_src: bool = False) -> None:
    """Copy images from source dataset to output."""
    for img_lst in tqdm(imgs.values()):
        for ip in img_lst:
            img_folder = os.path.split(os.path.split(ip)[0])[-1]
            img_name = os.path.split(ip)[-1]
            img_out_path = os.path.join(output_fp, img_folder, img_name)
            shutil.copy(ip, img_out_path)
            if del_from_src:
                os.remove(ip)


def imgs_distr(
    ds_path: str,
    index_func: Callable[[tuple], tuple],
    xlim: Tuple[int, int] = None
) -> None:
    """Prepare data and plot images amount distribution for models."""
    distr_data = get_imgs_distr_data(ds_path, index_func)
    plot_distr(distr_data, xlim=xlim)


def get_imgs_distr_data(
        ds_path: str,
        index_func: Callable[[tuple], tuple]
) -> list:
    """Prepare data to plot images amount distribution."""
    models_counter = objects_counter(ds_path, index_func)
    res = list(models_counter.values())
    return res


def objects_counter(
        ds_path: str,
        index_func: Callable[[tuple], tuple]
) -> defaultdict:
    """Get objects counter due to amount of images."""
    folders = os.listdir(ds_path)
    indexes = [index_func(indx) for indx in indexing.get_indexes(folders)]
    counter = defaultdict(int)
    iterator = tqdm(zip(folders, indexes), total=len(folders))
    for fold, indx in iterator:
        imgs_loc_path = os.path.join(ds_path, fold)
        imgs_len = len(os.listdir(imgs_loc_path))
        counter[indx] += imgs_len
    return counter


def plot_distr(distr_data: list, xlim: Tuple[int, int] = None):
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


def filter_objects_by_th(
    ds_path: str,
    index_func: Callable[[tuple], tuple],
    th: int
) -> dict:
    """Filter objects in dataset by threshold."""
    counter = objects_counter(ds_path, index_func)
    to_del = {k: v for k, v in filter(lambda x: x[1] < th, counter.items())}
    user_answer = check_del(to_del)
    if user_answer:
        objects_to_del = list(to_del.keys())
        delete_objects(ds_path, objects_to_del, index_func)
    return to_del


def check_del(to_del: dict) -> bool:
    """Check is there files to delete."""
    if to_del:
        user_answer = user_call_to_del(to_del)
        return user_answer
    else:
        print('Theres nothing to delete.')
        return False


def user_call_to_del(to_del: dict) -> bool:
    """User check to delete found files."""
    text = 'Following objects will be deleted from the dataset. Delete? [y/n]: '
    for obj, cnts in to_del.items():
        text = text + f'\n  {obj}: {cnts} imgs'
    while True:
        inp = input(f'{text}\n')
        if inp == 'y':
            return True
        elif inp == 'n':
            return False
        else:
            print('type either "y" or "n".')


def delete_objects(
    ds_path: str,
    indexes: list,
    index_func: Callable[[tuple], tuple]
) -> None:
    """Delete objects in dataset that """
    for index in indexes:
        del_by_index(ds_path, index, index_func)
        print(f'Object "{index}" has been deleted from dataset.')


def del_by_index(
    ds_path: str,
    index_to_del: Tuple,
    index_func: Callable[[tuple], tuple]
) -> None:
    """Delete dataset folders that have the given index."""
    folders = os.listdir(ds_path)
    indexes_t = [index_func(indx) for indx in indexing.get_indexes(folders)]
    for fold, indx in zip(folders, indexes_t):
        if indx == index_to_del:
            fold_path = os.path.join(ds_path, fold)
            delete_folder(fold_path)


def delete_folder(folder_path: str) -> None:
    """Delete given folder."""
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
    os.rmdir(folder_path)


def get_test_ds_from_train(
        train_ds_path: str,
        test_size: float,
        output_path: str,
        out_folder_name: str,
        random_state: int = None
) -> dict:
    """Replace sampled images from train to test dataset."""
    def marks_indx_f(x):
        return x[0], x[1]

    folders = os.listdir(train_ds_path)
    indexes = indexing.get_indexes(folders)
    collected_ds = collect_images(train_ds_path, folders, indexes, marks_indx_f)
    chosen_to_test = sample_to_test(collected_ds, test_size, random_state)
    make_images_copy(chosen_to_test, output_path,
                     out_folder_name, del_from_src=True)
    return chosen_to_test


def sample_to_test(
    collected_imgs: dict,
    test_size: float,
    random_state: int
) -> dict:
    """Sample images to create test dataset."""
    chosen_to_test = {}
    for mark, imgs in collected_imgs.items():
        train_size = len(imgs)
        test_size_int = ceil(test_size * train_size)
        hrs.handle_random_state(random_state)
        chosen_to_test[mark] = random.sample(imgs, test_size_int)
    return chosen_to_test
