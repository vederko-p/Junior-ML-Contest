
from typing import List

from tqdm.notebook import tqdm
import numpy as np
import torch

from dataset.utils.image_process import open_image
import dataset.default_augmentations as def_augs
from tools.feature_base import collect_dataset_total as collect_ds_t
from model_selection.metrics import CustomAccuracy, F1Score
from tools.vectorizer import Vectorizer


def eval_classif_paths(
        paths: List[str],
        model: torch.nn.Module,
) -> torch.tensor:
    pred_lbls = torch.empty(0, dtype=torch.long)
    model.eval()
    for ip in tqdm(paths):
        img_array = open_image(ip)
        img_array = def_augs.TL_no_aug_transform_128(image=img_array)['image']
        img_tens = torch.permute(torch.FloatTensor(img_array), (2, 0, 1))
        with torch.no_grad():
            pred_p = model.forward(img_tens.unsqueeze(0))
        pred_lbls = torch.cat([pred_lbls, pred_p.argmax(axis=1)])
    return pred_lbls


def test_classification(ds_path: str, tl_model, metric_model,
                        crop: bool = False):
    # Подготовка:
    imgs_pths, _, lbls_mrks, _, lbls_mdls, _ = collect_ds_t(ds_path)
    vect = Vectorizer()
    vect.vectorize_from_paths(imgs_pths, tl_model, crop)
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


def test_classification_splited(ds_path: str, tl_model, metric_model,
                                view_model, truck_model, crop: bool = False):
    # Подготовка:
    imgs_pths, _, lbls_mrks, _, lbls_mdls, _ = collect_ds_t(ds_path)
    vect = Vectorizer()
    vect.vectorize_from_paths(imgs_pths, tl_model, crop)
    keys = {
        'View': eval_classif_paths(imgs_pths, view_model).squeeze().tolist(),
        'Truck': eval_classif_paths(imgs_pths, truck_model).squeeze().tolist()
    }
    x = vect.vects_from_paths_res.numpy().astype(np.float32)
    mrk_res, _, mdl_res, _ = metric_model.forward(x, keys=keys)
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


def test_classification_joint(ds_path: str, tl_marks_model, tl_models_model,
                              metric_model, view_model, truck_model):
    # Подготовка:
    imgs_pths, _, lbls_mrks, _, lbls_mdls, _ = collect_ds_t(ds_path)
    # Вычисление признаков:
    vectorizer_new = Vectorizer()
    vectorizer_new.vectorize_from_paths(imgs_pths, tl_marks_model)
    vectorizer_old = Vectorizer()
    vectorizer_old.vectorize_from_paths(imgs_pths, tl_models_model, crop=True)
    vectorizer_joint = Vectorizer()
    vectorizer_joint.vects_from_paths_res = torch.cat(
        [vectorizer_new.vects_from_paths_res,
         vectorizer_old.vects_from_paths_res],
        axis=1)
    # Ракурс / Грузовик:
    keys = {
        'View': eval_classif_paths(imgs_pths, view_model).squeeze().tolist(),
        'Truck': eval_classif_paths(imgs_pths, truck_model).squeeze().tolist()
    }
    # Распознавание:
    x = vectorizer_joint.vects_from_paths_res.numpy().astype(np.float32)
    mrk_res, _, mdl_res, _ = metric_model.forward(x, keys=keys)
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


def test_distance(imgs_pths, tl_marks_model, tl_models_model,
                  metric_model, view_model, truck_model):
    # Вычисление признаков:
    vectorizer_new = Vectorizer()
    vectorizer_new.vectorize_from_paths(imgs_pths, tl_marks_model)
    vectorizer_old = Vectorizer()
    vectorizer_old.vectorize_from_paths(imgs_pths, tl_models_model, crop=True)
    vectorizer_joint = Vectorizer()
    vectorizer_joint.vects_from_paths_res = torch.cat(
        [vectorizer_new.vects_from_paths_res,
         vectorizer_old.vects_from_paths_res],
        axis=1)
    # Ракурс / Грузовик:
    keys = {
        'View': eval_classif_paths(imgs_pths, view_model).squeeze().tolist(),
        'Truck': eval_classif_paths(imgs_pths, truck_model).squeeze().tolist()
    }
    # Распознавание:
    x = vectorizer_joint.vects_from_paths_res.numpy().astype(np.float32)
    _, mrk_dist, _, mdl_dist = metric_model.forward(x, keys=keys)
    return mrk_dist, mdl_dist
