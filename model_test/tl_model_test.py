
from typing import Tuple

import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

from dataset.tl_dataset_test import show_triplet
from models.triplet_loss import TripletLossModel
from dataset.custom_datasets import TripletLossDataset


def test_tl_moldel(model: TripletLossModel,
                   ds: TripletLossDataset,
                   n_rows: int) -> plt.figure:
    """Test TL model on multiple random triplets."""
    block_rows = 3
    n_cols = 3
    fig = plt.figure()
    gs_base = fig.add_gridspec(n_rows, 1, wspace=-1, hspace=-1, figure=fig)
    for _ in range(n_rows):
        gsi = GridSpecFromSubplotSpec(block_rows, n_cols,
                                      subplot_spec=gs_base[0])
        _ = tl_model_test_1d(model, ds, show=False, gs=gsi)
    plt.show()
    return fig


def tl_model_test_1d(model: TripletLossModel,
                     ds: TripletLossDataset,
                     line_size: int = 10,
                     show: bool = True,
                     gs=None):
    """Test TL model on one random triplet."""
    n_cols = 3
    image_size = 128
    block_rows = 3
    block_width = 10
    figsize = (block_width, line_size)
    fig = plt.figure(figsize=figsize)
    if gs is None:
        gs = plt.GridSpec(block_rows, n_cols)
    axs = []
    triplet = ds.__getitem__(0, img_meta=True)
    # Векторизация:
    triplet_batch = torch.empty((3, 1, image_size, image_size), dtype=torch.float)
    for ik, k in enumerate(['Anchor', 'Positive', 'Negative']):
        triplet_batch[ik] = triplet[k]
    with torch.no_grad():
        pred_vect = model.forward(triplet_batch)
    # Визуализация триплета:
    triplet_axes = [fig.add_subplot(gs[0, j]) for j in range(n_cols)]
    axs.append(triplet_axes)
    _ = show_triplet(triplet, axs=triplet_axes, show=False, title_fs=12)
    # Визуализация векторизации Anchor-Positive:
    axs.append(fig.add_subplot(gs[1, :]))
    _ = plot_vector(pred_vect[0], ax=axs[-1], label='Anchor')
    _ = plot_vector(pred_vect[1], ax=axs[-1], label='Positive')
    # Визуализация векторизации Anchor-Negative:
    axs.append(fig.add_subplot(gs[2, :]))
    _ = plot_vector(pred_vect[0], ax=axs[-1], label='Anchor')
    _ = plot_vector(pred_vect[2], ax=axs[-1], label='Negative')
    if show:
        plt.show()
    return gs


def plot_vector(vect: torch.tensor,
                line_s: Tuple[int, int] = (8, 3),
                ax: plt.axes = None, label: str = None,
                show: bool = False) -> plt.axes:
    """Plot vector."""
    # vect -
    if label is None:
        label = ''
    if ax is None:
        fig, ax = plt.subplots(figsize=line_s)
    ax.plot(vect.numpy(), label=label)
    plt.legend()
    if show:
        plt.show()
    return ax
