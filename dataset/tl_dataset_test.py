
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt

from dataset.custom_datasets import TripletLossDataset


def show_part_of_triplet(img_tens: torch.tensor,
                         title: str = '',
                         ax: plt.axes = None,
                         show: bool = True,
                         title_fs: int = 12) -> plt.axes:
    """Show part of triplet."""
    if ax is None:
        fig, ax = plt.subplots()
    img_array = torch.permute(img_tens, (1, 2, 0)).numpy()
    ax.imshow(img_array, 'gray')
    ax.set_title(title, fontsize=title_fs)
    if show:
        plt.show()
    return ax


def show_triplet(triplet: dict, axs: List[plt.axes] = None,
                 fig_ln_size: Tuple[int, int] = (10, 3),
                 show: bool = True,
                 line_span: int = 1, title_fs: int = 12) -> List[plt.axes]:
    """Show triplet."""
    triplet_keys = ['Anchor', 'Positive', 'Negative']
    n_cols = 3
    if axs is None:
        actual_fig_size = (fig_ln_size[0], fig_ln_size[1]+line_span)
        fig, gs = plt.figure(figsize=actual_fig_size), plt.GridSpec(1, n_cols)
        axs = [fig.add_subplot(gs[0, j]) for j in range(n_cols)]
    for j, tl_k in enumerate(triplet_keys):
        title = f'({tl_k})\n' + triplet[f'{tl_k}_name']
        show_part_of_triplet(triplet[tl_k], title=title, ax=axs[j],
                             title_fs=title_fs, show=False)
    if show:
        plt.show()
    return axs


def tl_ds_test(ds: TripletLossDataset,
               n_rows: int, line_span: int = 1,
               fig_ln_size: Tuple[int, int] = (10, 3)) -> None:
    """Test triplet loss dataset."""
    n_cols = 3
    actual_fig_size = (fig_ln_size[0], (fig_ln_size[1]+line_span)*n_rows)
    fig = plt.figure(figsize=actual_fig_size)
    gs = plt.GridSpec(n_rows, n_cols, figure=fig)
    for row in range(n_rows):
        axs = [fig.add_subplot(gs[row*n_cols + i]) for i in range(n_cols)]
        triplet = ds.__getitem__(0, img_meta=True)
        _ = show_triplet(triplet, axs=axs, show=False,
                         fig_ln_size=fig_ln_size)
    plt.show()
