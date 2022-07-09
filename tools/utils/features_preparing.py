
from typing import List
from collections import defaultdict
from random import sample
from dataset.utils.handle_random_state import handle_random_state


def chose_objects(imgs_paths: List[str], lbls: List[int],
                  lbls_code: dict, n: int, k: int,
                  random_state: int = None) -> dict:
    """Randomly chose objects of same randomly chosen classes.

        Parameters
        ----------
        imgs_paths : `list`
            List of images paths.
        lbls: : `list`
            List of images labels.
        lbls_code : `dict`
            Labels code.
        n : `int`
            Amount of classes.
        k : `int`
            Amount of max images from each randomly chosen class.
        random_state : `int`
            Random state for classes and images sampling.

        Returns
        -------
        images_storage : `dict`
            Dict os sampled images: {mark / mark+model: images}.
        """
    handle_random_state(random_state)
    chosen_lbls = sample(list(lbls_code.values()), n)
    img_indexes = range(len(imgs_paths))
    t_img_indexes = filter(lambda i: lbls[i] in chosen_lbls, img_indexes)
    storage = defaultdict(list)
    for t_indx in t_img_indexes:
        storage[lbls[t_indx]].append(imgs_paths[t_indx])
    storage_out = {}
    for lbl, imgs in storage.items():
        handle_random_state(random_state)
        storage_out[lbl] = sample(imgs, min(len(imgs), k))
    return storage_out


def transform_chosen_objects(chosen_objects: dict):
    """Chosen images transformation to images and labels lists."""
    images = []
    labels = []
    for k, v in chosen_objects.items():
        labels.extend([k]*len(v))
        images.extend(v)
    return images, labels
