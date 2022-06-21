
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


class CustomAccuracy:
    def __init__(self):
        self.res_scope = None
        self.res_ds_scope = None
    
    def scope(self, y_pred, y_true):
        # y_pred, y_true - метки классов ~ torch.tensor
        res = torch.eq(y_pred, y_true).sum() / y_pred.shape[0]
        self.res_scope = res
        return res.item()
    
    def ds_scope(self, ds, model):
        l = len(ds)
        batch_size= 32
        dataloader = DataLoader(ds, batch_size=batch_size)
        res = 0
        model.eval()
        iterator = tqdm(dataloader)
        for batch in iterator:
            img_batch, lbl_batch = batch
            with torch.no_grad():
                model_pred_p = model.forward(img_batch)
            model_pred_lbl = model_pred_p.argmax(axis=1)
            bath_acc = torch.eq(model_pred_lbl, lbl_batch).sum().item()
            res += bath_acc
        res = res / l
        self.res_ds_scope = res
        return res


class CustomVarianceCriteria:
    def __init__(self):
        self.res = None
        
    def scope(self, X, y):
        # X, y - torch.tensor; dtype=torch.float
        unique_lbls = torch.unique(y)
        size = unique_lbls.shape[0]
        groups_mean = torch.empty(size, dtype=torch.float)
        groups_var = torch.empty(size, dtype=torch.float)
        for i,k in enumerate(unique_lbls):
            groups_mean[i] = X[y==k].mean()
            groups_var[i] = X[y==k].var()
        D_in = groups_var.mean().item()
        D_mg = groups_var.var().item()
        return D_in / D_mg
