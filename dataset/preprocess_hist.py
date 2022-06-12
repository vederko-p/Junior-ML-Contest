
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Callable


def get_unique(indexes: list, indx_func: Callable[[tuple], tuple]) -> Tuple[list, set, dict]:
    target_indexes = [indx_func(indx) for indx in indexes]
    unique = set(target_indexes)
    rng = range(len(unique))
    code_in = {instnc: cd for instnc, cd in zip(unique, rng)}
    return target_indexes, unique, code_in


def get_hist_data(ds_path: str, folders: list, t_indexes: list, code_in: dict):
    hist_data = []
    iterator = tqdm(zip(folders, t_indexes), total=len(folders))
    for fold, indx in iterator:
        imgs_path = os.path.join(ds_path, fold)
        imgs_len = len(os.listdir(imgs_path))
        hist_data.extend([code_in[indx]]*imgs_len)
    return hist_data


def plot_hist(hist_data, unique, title, xlabel, ylim: Tuple[int, int] = None) -> None:
    bins = range(len(unique))
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel('Число изображений', fontsize=13)
    ax.hist(hist_data, bins=bins)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.show()
