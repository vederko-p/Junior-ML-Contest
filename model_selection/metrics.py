
from typing import Tuple

from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


def eval_classif_ds(ds: Dataset,
            model: torch.nn.Module) -> Tuple[torch.tensor, torch.tensor]:
    batch_size = 32
    true_lbls = torch.empty(0, dtype=torch.long)
    pred_lbls = torch.empty(0, dtype=torch.long)
    model.eval()
    dataloader = DataLoader(ds, batch_size=batch_size)
    for batch in tqdm(dataloader):
        img_batch, lbl_batch = batch
        with torch.no_grad():
            pred_p = model.forward(img_batch)
        true_lbls = torch.cat([true_lbls, lbl_batch])
        pred_lbls = torch.cat([pred_lbls, pred_p.argmax(axis=1)])
    return true_lbls, pred_lbls


class CustomAccuracy:
    """Torch accuracy implementation"""
    def __init__(self):
        self.res_scope = None
        self.res_ds_scope = None
    
    def scope(self, y_true: torch.tensor, y_pred: torch.tensor) -> float:
        """Evaluate accuracy over tensor data.

        Parameters
        ----------
        y_pred : `torch.tensor`
            Class predictions by model.
        y_true : `torch.tensor`
            True classes.

        Returns
        -------
        acc : `float`
            Accuracy.
        """
        res = torch.eq(y_true, y_pred).sum() / y_pred.shape[0]
        self.res_scope = res
        return res.item()

    def ds_scope(self, ds: Dataset, model: torch.nn.Module):
        """Evaluate accuracy over dataset.

        Parameters
        ----------
        ds : `torch.utils.data.Dataset`
            Classify dataset.
        model : `torch.tensor`
            Model.

        Returns
        -------
        acc : `float`
            Accuracy.
        """
        y_true, y_pred = eval_classif_ds(ds, model)
        res = self.scope(y_true, y_pred)
        self.res_ds_scope = res
        return res


class CustomVarianceCriteria:
    """Torch variance criteria implementation"""
    def __init__(self):
        self.res_scope = None
        
    def scope(self, x: torch.tensor, y: torch.tensor) -> float:
        """Evaluate accuracy over tensor data.

        Parameters
        ----------
        x : `torch.tensor`
            Tensor of vectors with dtype=torch.float.
        y : `torch.tensor`
            Class label.

        Returns
        -------
        acc : `float`
            Accuracy.
        """
        unique_lbls = torch.unique(y)
        size = unique_lbls.shape[0]
        groups_mean = torch.empty(size, dtype=torch.float)
        groups_var = torch.empty(size, dtype=torch.float)
        for i, k in enumerate(unique_lbls):
            groups_mean[i] = x[y == k].mean()
            groups_var[i] = x[y == k].var()
        D_in = groups_var.mean().item()
        D_mg = groups_var.var().item()
        crit_val = D_in / D_mg
        self.res_scope = crit_val
        return crit_val


class F1Score:
    """Torch f1 score implementation"""
    def __init__(self):
        self.res_scope = None
        self.res_ds_scope = None

    def scope(self, y_true: torch.tensor, y_pred: torch.tensor) -> float:
        res = f1_score(y_true, y_pred, average='weighted')
        self.res_scope = res
        return res

    def ds_scope(self, ds: Dataset, model: torch.nn.Module) -> float:
        y_true, y_pred = eval_classif_ds(ds, model)
        res = self.scope(y_true, y_pred)
        self.res_ds_scope = res
        return res
