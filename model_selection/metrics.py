
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


class CustomAccuracy:
    """Torch accuracy implementation"""
    def __init__(self):
        self.res_scope = None
        self.res_ds_scope = None
    
    def scope(self, y_pred: torch.tensor, y_true: torch.tensor) -> float:
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
        res = torch.eq(y_pred, y_true).sum() / y_pred.shape[0]
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
    """Torch variance criteria implementation"""
    def __init__(self):
        self.res_scope = None
        
    def scope(self, x: torch.tensor, y: torch.tensor):
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

    def scope(self):
        pass

    def ds_scope(self):
        pass
