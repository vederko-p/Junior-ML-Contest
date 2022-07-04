
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import dataset.utils.image_process as img_proc
from dataset import default_augmentations as def_augs


class Vectorizer:
    def __init__(self):
        self.lables_res = None
        self.vects_res = None
        self.vects_from_paths_res = None

    def vectorize(self, ds, model, batch_size: int = 32) -> None:
        model.eval()
        labels = torch.empty(0, dtype=torch.long)
        vectors = torch.empty((0, model.n_out), dtype=torch.float)
        dataloader = DataLoader(ds, batch_size=batch_size)
        for batch in tqdm(dataloader):
            imgs, lbls = batch
            with torch.no_grad():
                vects = model.forward(imgs)
            labels = torch.cat([labels, lbls])
            vectors = torch.cat([vectors, vects])
        self.lables_res = labels
        self.vects_res = vectors

    def vectorize_from_paths(self, imgs_paths: List[str], model) -> None:
        model.eval()
        vectors = torch.empty((0, model.n_out), dtype=torch.float)
        for i, ip in enumerate(tqdm(imgs_paths)):
            img_array = img_proc.open_image(ip)
            img_array = def_augs.TL_no_aug_transform_128(image=img_array)['image']
            img_tens = torch.permute(torch.FloatTensor(img_array), (2, 0, 1))
            with torch.no_grad():
                vect = model.forward(img_tens.unsqueeze(0))
            vectors = torch.cat([vectors, vect.unsqueeze(0)])
        self.vects_from_paths_res = vectors
