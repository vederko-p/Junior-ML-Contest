
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Callable


def get_unique(
        indexes: list,
        index_func: Callable[[tuple], tuple]
) -> Tuple[list, set, dict]:
    """Get unique objects.

    Parameters
    ----------
    indexes : `list`
        List of dataset folders indexes.
    index_func : `Callable`
        Function that define the folder index slice method to describes object
        index.

    Returns
    -------
    target_indexes : `list`
        List of indexes processed by index function.
    unique : `set`
        Set of unique objects indexes due to index function.
    code_in : `dict`
        Unique objects code.
    """
    target_indexes = [index_func(indx) for indx in indexes]
    unique = set(target_indexes)
    rng = range(len(unique))
    code_in = {instnc: cd for instnc, cd in zip(unique, rng)}
    return target_indexes, unique, code_in


def get_hist_data(
        ds_path: str,
        folders: list,
        t_indexes: list,
        code_in: dict
) -> list:
    """Get histogram format data on images.

    Parameters
    ----------
    ds_path : `str`
        Dataset path.
    folders : `list`
        List of folders names.
    t_indexes : `list`
        Objects indexes.
    code_in : `dict`
        Objects indexes code.

    Returns
    -------
    hist_data : `list`
        List of histogram data format.
    """
    hist_data = []
    iterator = tqdm(zip(folders, t_indexes), total=len(folders))
    for fold, indx in iterator:
        imgs_path = os.path.join(ds_path, fold)
        imgs_len = len(os.listdir(imgs_path))
        hist_data.extend([code_in[indx]]*imgs_len)
    return hist_data


def plot_hist(
        hist_data: list,
        unique: set,
        title: str,
        xlabel: str,
        ylim: Tuple[int, int] = None
) -> None:
    """Plots histogram."""
    bins = range(len(unique)+1)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel('Число изображений', fontsize=13)
    ax.hist(hist_data, bins=bins)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.show()
