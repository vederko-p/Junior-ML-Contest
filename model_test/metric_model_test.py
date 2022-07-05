
import numpy as np
import torch

from tools.feature_base import collect_dataset_total as collect_ds_t
from model_selection.metrics import CustomAccuracy, F1Score
from tools.vectorizer import Vectorizer


def test_classification(ds_path: str, tl_model, metric_model):
    # Подготовка:
    imgs_pths, _, lbls_mrks, _, lbls_mdls, _ = collect_ds_t(ds_path)
    vect = Vectorizer()
    vect.vectorize_from_paths(imgs_pths, tl_model)
    x = vect.vects_from_paths_res.numpy().astype(np.float32)
    mrk_res, _, mdl_res, _ = metric_model.forward(x)
    # Accuracy:
    marks_acc = CustomAccuracy().scope(torch.tensor(lbls_mrks),
                                       torch.tensor(mrk_res))
    models_acc = CustomAccuracy().scope(torch.tensor(lbls_mdls),
                                        torch.tensor(mdl_res))
    # f1:
    marks_f1 = F1Score().scope(torch.tensor(lbls_mrks),
                               torch.tensor(mrk_res))
    models_f1 = F1Score().scope(torch.tensor(lbls_mdls),
                                torch.tensor(mdl_res))
    return marks_acc, models_acc, marks_f1, models_f1
