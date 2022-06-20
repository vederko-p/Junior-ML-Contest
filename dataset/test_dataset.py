
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from dataset.classify_dataset import TLClassifyDataset
import dataset.utils.handle_random_state as hrs


def classify_ds_test(ds: TLClassifyDataset,
                     n: int, figsize_sq: int = 3,
                     add_to_title: list = None,
                     lines_span: int = 1,
                     imgs_indexes: list = None) -> None:
    """Test dataset.

    Parameters
    ----------
    ds : `TLClassifyDataset`
        Dataset instance.
    n : `int`
        Amount of rows.
    figsize_sq : `int`
        Size of one square img.
    add_to_title : `list`
        List of additional strings to the titles of squares. Must match with
        shape of plot in squares.
    lines_span : `int`
        Span between lines.
    imgs_indexes : `list`
        List of images indexes due to dataset instance. Must match with n_cols
        which is 3 by default.
    """
    n_cols = 3
    figsize_ln = ((figsize_sq+1)*n_cols, (figsize_sq+lines_span)*n)
    figsize_sq = (figsize_sq, figsize_sq)
    fig, gs = plt.figure(figsize=figsize_ln), plt.GridSpec(n, n_cols)
    axs = []
    if add_to_title is None:
        add_to_title_line = [None] * n_cols
        add_to_title = [add_to_title_line] * n
    else:
        if (len(add_to_title) != n) and (len(add_to_title[0]) != n_cols):
            raise Exception('Shape of arg add_to_title must'
                            'be equal to (n, 3).')
    if imgs_indexes is None:
        imgs_indexes = get_indexes_for_vis(ds, n)
    else:
        if (len(imgs_indexes) != n) and (len(imgs_indexes[0]) != n_cols):
            raise Exception('Shape of arg "imgs_indexes" must'
                            'be equal to (n, 3).')
    for r in range(n*n_cols):
        axs.append(fig.add_subplot(gs[r]))
        row = r // n_cols
        col = r % n_cols
        show_img(ds, imgs_indexes[row][col], figsize=figsize_sq,
                 ax=axs[-1], add_to_title=add_to_title[row][col])


def get_indexes_for_vis(ds, n, random_state=None):
    hrs.handle_random_state(random_state)
    objects = random.sample(list(set(ds.labels)), n)
    iterator = list(zip(ds.labels, range(len(ds.labels))))
    imgs_in_row = 3
    imgs_indexes = []
    for obj_j in objects:
        obj_filtred = list(filter(lambda x: x[0] == obj_j, iterator))
        hrs.handle_random_state(random_state)
        imgs_t = random.sample(obj_filtred, imgs_in_row)
        ds_indxs = list(map(lambda x: x[1], imgs_t))
        imgs_indexes.append(ds_indxs)
    return imgs_indexes


def show_img(ds: TLClassifyDataset, indx: int,
             figsize: Tuple[int, int] = (3, 3),
             ax=plt.axes, add_to_title: str = None) -> plt.axes:
    tens_img, tens_lbl = ds[indx]
    lbl_itm = tens_lbl.item()
    labels_code_out = {v: k for k, v in ds.labels_code.items()}
    lbl_name = labels_code_out[lbl_itm]
    array_img = np.transpose(tens_img.numpy(), axes=(1, 2, 0))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(array_img, 'gray')
    if add_to_title is not None:
        add_to_title = f'\n{add_to_title}'
    else:
        add_to_title = ''
    ax.set_title(f'{lbl_name} ({lbl_itm})' + add_to_title, fontsize=15)
    return ax
