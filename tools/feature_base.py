
import os
from typing import List

from dataset import indexing
from tools.vectorizer import Vectorizer


def float_q(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def marks_indx_f(x):
    return (x[0],)


def models_indx_f(x):
    return (x[0], x[1])


def collect_dataset_total(ds_path: str) -> tuple:
    images, indexes_images = [], []
    labels_marks, labels_models = [], []
    labels_marks_code, labels_models_code = {}, {}
    folders = os.listdir(ds_path)
    indexes_folders = indexing.get_indexes(folders)
    for fold, indx in zip(folders, indexes_folders):
        mark_indx = marks_indx_f(indx)
        if mark_indx not in labels_marks_code.keys():
            labels_marks_code[mark_indx] = len(labels_marks_code.keys())

        model_indx = models_indx_f(indx)
        if model_indx not in labels_models_code.keys():
            labels_models_code[model_indx] = len(labels_models_code.keys())

        imgs_path = os.path.join(ds_path, fold)
        imgs_loc_paths = os.listdir(imgs_path)
        imgs_glob_paths = [os.path.join(imgs_path, ilp) for ilp in imgs_loc_paths]
        images.extend(imgs_glob_paths)

        labels_marks.extend([labels_marks_code[mark_indx]]*len(imgs_glob_paths))
        labels_models.extend([labels_models_code[model_indx]]*len(imgs_glob_paths))
        indexes_images.extend([indx]*len(imgs_glob_paths))
    res = (images, indexes_images,
           labels_marks, labels_marks_code,
           labels_models, labels_models_code)
    return res


def save_table(data: dict, filepath: str) -> None:
    if os.path.isfile(filepath):
        os.remove(filepath)
    cols = list(data.keys())
    cols_w = ','.join(cols) + '\n'
    with open(filepath, 'w') as output_file:
        output_file.writelines([cols_w])
        for i in data['ID']:
            line = ','.join([' '.join(data[col][i])
                             if isinstance(data[col][i], tuple)
                             else str(data[col][i]) for col in cols]) + '\n'
            output_file.writelines([line])


def load_table(filepath: str) -> dict:
    with open(filepath, 'r') as input_file:
        data_lines = input_file.readlines()
    cols = data_lines[0][:-1].split(',')
    data = {col: [] for col in cols}
    for line in data_lines[1:]:
        vals_str = line[:-1].split(',')
        vals = [(int(v) if str.isdigit(v) else float(v)) if float_q(v)
                else tuple(v.split(' ')) for v in vals_str]
        for col, val in zip(cols, vals):
            data[col].append(val)
    return data


class FeaturesBase:
    def __init__(self, features_len: int):
        self.features_len = features_len
        self.data_marks = None
        self.data_models = None
        self.data_all = None
        self.vect = None
        self.make_empty_base()

    def make_empty_base(self):
        self.vect = Vectorizer()
        self.data_marks = {'ID': [], 'ID_mark': [], 'Name': []}
        self.data_models = {'ID': [], 'ID_model': [], 'Name': []}
        self.data_all = {'ID': [],
                         'ID_mark': [], 'ID_model': [],
                         'View': [], 'Truck': []}
        for i in range(self.features_len):
            self.data_all[f'f_{i}'] = []

    def fill_from_path(self, ds_path: str, model, crop: bool = False,
                       custom_vectorizer: Vectorizer = None):
        if self.vect is None:
            raise Exception('You must create empty base first.')

        c_ds = collect_dataset_total(ds_path)
        images, indexes = c_ds[0], c_ds[1]
        lbls_mrks, lbls_mrks_code = c_ds[2], c_ds[3]
        lbls_mdls, lbls_mdls_code = c_ds[4], c_ds[5]

        self.fill_marks_data(lbls_mrks_code)
        self.fill_models_data(lbls_mdls_code)
        self.fill_all_data(images, indexes,
                           lbls_mrks, lbls_mdls, model,
                           crop, custom_vectorizer)

    def fill_marks_data(self, marks_labels_code: dict) -> None:
        for mrk_n, mrk_id in marks_labels_code.items():
            self.data_marks['ID'].append(len(self.data_marks['ID']))
            self.data_marks['ID_mark'].append(mrk_id)
            self.data_marks['Name'].append(mrk_n)

    def fill_models_data(self, models_labels_code: dict) -> None:
        for mdl_n, mdl_id in models_labels_code.items():
            self.data_models['ID'].append(len(self.data_models['ID']))
            self.data_models['ID_model'].append(mdl_id)
            self.data_models['Name'].append(mdl_n)

    def fill_all_data(self,
                      imgs: List[str], indxs: List[tuple],
                      marks_labels: List[int],
                      models_labels: List[int],
                      model, crop: bool = False,
                      custom_vectorizer: Vectorizer = None) -> None:
        if custom_vectorizer is None:
            self.vect.vectorize_from_paths(imgs, model, crop)
        else:
            self.vect = custom_vectorizer
        iterator = enumerate(zip(indxs, marks_labels, models_labels))
        for i, (indx, mrk_lbl, mdl_lbl) in iterator:
            view = 1 if indx[2] == 'normal' else 0
            truck = 1 if indx[3] == 'truck' else 0
            features = self.vect.vects_from_paths_res[i].tolist()
            self.append_row(mrk_lbl, mdl_lbl, view, truck, features)

    def append_row(self, mark_lbl, model_lbl, view, truck, ftrs) -> None:
        self.data_all['ID'].append(len(self.data_all['ID']))
        self.data_all['ID_mark'].append(mark_lbl)
        self.data_all['ID_model'].append(model_lbl)
        self.data_all['View'].append(view)
        self.data_all['Truck'].append(truck)
        for i, ftr in enumerate(ftrs):
            self.data_all[f'f_{i}'].append(ftr)

    def save_base(self, filepaths: List[str]) -> None:
        if not self.data_marks['ID']:
            raise Exception('You must get some data first.')
        total_data = [self.data_marks, self.data_models, self.data_all]
        for data, fn in zip(total_data, filepaths):
            save_table(data, fn)

    def load_base(self, filepaths: List[str]) -> None:
        self.data_marks = load_table(filepaths[0])
        self.data_models = load_table(filepaths[1])
        self.data_all = load_table(filepaths[2])
