
from typing import Tuple


def get_index(folder_name: str) -> Tuple[str, str, str, str]:
    """Returns index of folder by its name."""
    split_name = folder_name.split(' ')
    mark, model = split_name[0], split_name[1]
    view = 'back' if 'back' in folder_name else 'normal'
    truck = 'truck' if 'truck' in folder_name else 'no_truck'
    index = (mark, model, view, truck)
    return index


def get_indexes(folders: list) -> list:
    """Returns list of folders indexes."""
    indexes = [get_index(fold_n) for fold_n in folders]
    return indexes
