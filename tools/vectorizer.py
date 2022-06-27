
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


class Vectorizer:
    def __init__(self):
        self.lables_res = None
        self.vects_res = None

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
