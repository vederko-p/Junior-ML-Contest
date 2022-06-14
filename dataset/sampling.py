
import os
import shutil
import random
from math import ceil
from typing import Tuple
from tqdm.notebook import tqdm
from collections import defaultdict
from dataset import indexing


def sample_data(ds_path: str,
                output_path: str,
                min_th: int = 1000,
                max_th: int = None,
                random_state: int = None) -> dict:
    """Random sample images from dataset by marks.

    Parameters
    ----------
    ds_path : `str`
        Dataset path.
    output_path : `str`
        Path for sampled dataset output.
    min_th, max_th : `int`, `int`
        Minimal and maximal thresholds values of images amount to sample.
    random_state : `int`
        Random state for sampler.

    Returns
    -------
    sampled_images : `dict`
        Dict os sampled images: {mark: images}.
    """
    folders = os.listdir(ds_path)
    indexes = indexing.get_indexes(folders)
    imgs_by_marks = collect_imgs_by_marks(ds_path, folders, indexes)
    sampled_imgs_by_marks = sample_imgs_bm(imgs_by_marks,
                                           min_th, max_th,
                                           random_state)
    user_answer = check_size(sampled_imgs_by_marks)
    if user_answer:
        make_images_copy(sampled_imgs_by_marks, output_path)
    return sampled_imgs_by_marks


def collect_imgs_by_marks(ds_path: str,
                          folders: list,
                          indexes: list) -> defaultdict:
    """Collect images by marks from dataset."""
    imgs_by_marks = defaultdict(list)
    iterator = tqdm(zip(folders, indexes), total=len(folders))
    for fold_n, indx in iterator:
        imgs_loc_path = os.path.join(ds_path, fold_n)
        imgs_paths = os.listdir(imgs_loc_path)
        imgs_gl_paths = [os.path.join(imgs_loc_path, ilp) for ilp in imgs_paths]
        imgs_by_marks[indx[0]].extend(imgs_gl_paths)
    return imgs_by_marks


def sample_imgs_bm(imgs_by_marks: dict,
                   min_th: int = 1000,
                   max_th: int = None,
                   random_state: int = None) -> defaultdict:
    """Sample images within each mark."""
    sampled_imgs = defaultdict(list)
    for mark, imgs in imgs_by_marks.items():
        imgs_len = len(imgs)
        if imgs_len > min_th:
            q = min(imgs_len, max_th) if max_th is not None else imgs_len
            handle_random_state(random_state)
            sampled_imgs[mark].extend(random.sample(imgs, q))
    return sampled_imgs


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
    text = f'{s} {tag} of free space are needed to complete sampling.' \
           f'Sample? [y/n]: '
    while True:
        inp = input(text)
        if inp == 'y':
            return True
        elif inp == 'n':
            return False
        else:
            print('type either "y" or "n".')


def make_images_copy(imgs: dict, output_path: str) -> None:
    """Copy folders and images into new dataset folder."""
    folders = collect_folders(imgs)
    output_folderpath = os.path.join(output_path, 'data_sampled')
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


def create_folders(output_fp: str, folders: list) -> None:
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
