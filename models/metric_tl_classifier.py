
from typing import Tuple, List
from random import sample

import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
import cv2 as cv

from tools.feature_base import FeaturesBase


def quasy_inv_matr(m, eps=10**(-10)):
    U, S, V = np.linalg.svd(m)
    dd = np.diag(np.diag(S))
    S = np.diag(S)
    dd2 = np.multiply(np.sign(dd), np.maximum(eps, np.abs(dd)))
    dd_ = np.divide(1, dd2 + np.finfo(float).eps)
    S_ = np.diag(dd_)
    S1 = np.transpose(S)
    S1[0:S_.shape[1]][0:S_.shape[0]] = S_
    return np.matmul(np.matmul(V.T, S1), U.T)


def inv_conv_matr(m):
    V = np.cov(m, rowvar=False)
    VI = quasy_inv_matr(V, eps=1)
    return VI


def mahalanobis(x, Y, VI):
    a = np.dot((x - Y), VI)
    b = (x - Y).T
    d = np.sqrt(np.einsum('ij,ji->i', a, b))
    return np.abs(np.mean(d))


class MarkskNN:
    def __init__(self, ftrs_base: FeaturesBase, keys: List[str] = None):
        self.df_all = pd.DataFrame(ftrs_base.data_all)
        self.keys = [] if keys is None else keys
        self.ftrs_len = 128
        self.knn = {}

    def forward(self, x, k_neighs=15, keys: dict = None):
        if self.keys:
            res = np.array([]).reshape(0, 1)
            dist = np.array([]).reshape(0, 1)
            for i in range(x.shape[0]):
                knn_key = tuple([keys[kn][i] for kn in self.keys])
                _, res_i, _, dist_i = self.knn[knn_key].findNearest(
                    x[i:i+1, :self.ftrs_len], k_neighs)
                res = np.vstack([res, res_i])
                dist = np.vstack([dist, dist_i[:, 0]])
        else:
            _, res, _, dist = self.knn[0].findNearest(x[:, :self.ftrs_len],
                                                      k_neighs)
        return res[:, 0].astype(np.int32), dist[:, 0]

    def set_knn(self):
        if self.keys:
            grouped = self.df_all.groupby(self.keys)
            for gk in grouped.groups:
                group = grouped.get_group(gk)
                rng = [f'f_{i}' for i in range(self.ftrs_len)]
                gX = group[rng].values.astype(np.float32)
                gy_marks = group['ID_mark'].values.astype(np.float32)
                self.knn[gk] = cv.ml.KNearest_create()
                self.knn[gk].train(gX, cv.ml.ROW_SAMPLE, gy_marks)
        else:
            X = self.df_all[
                [f'f_{i}' for i in range(self.ftrs_len)]
            ].values.astype(np.float32)
            y_marks = self.df_all['ID_mark'].values.astype(np.float32)
            self.knn[0] = cv.ml.KNearest_create()
            self.knn[0].train(X, cv.ml.ROW_SAMPLE, y_marks)


class ModelskNN:
    def __init__(self, ftrs_base: FeaturesBase, keys: List[str] = None):
        self.df_all = pd.DataFrame(ftrs_base.data_all)
        self.keys = [] if keys is None else keys
        self.mrk_ftrs_len = 128
        self.mdl_ftrs_len = 64
        self.knn = {}  # {mark_id[, *keys]: knn}

    def forward(self, x, marks_res_0, k_neighs=15, keys: dict = None):
        models_res = []
        models_dist = []
        if self.keys:
            ks = [keys[kn] for kn in self.keys]
            marks_res = zip(marks_res_0, *ks)
        else:
            marks_res = marks_res_0
        for i, mrk_id in enumerate(marks_res):
            try:
                _, res, _, dist = self.knn[mrk_id].findNearest(
                    x[i:i+1, -self.mdl_ftrs_len:], k_neighs)
            except KeyError:
                knn_key = list(filter(lambda u: u[0] == mrk_id[0], self.knn.keys()))[0]
                _, res, _, dist = self.knn[knn_key].findNearest(
                    x[i:i+1, -self.mdl_ftrs_len:], k_neighs)
            models_res.append(res[0, 0])
            models_dist.append(dist[0, 0])
        return np.array(models_res, dtype=np.int32), np.array(models_dist)

    def set_knn(self):
        grouped = self.df_all.groupby(['ID_mark'] + self.keys)
        for gk in grouped.groups:
            group = grouped.get_group(gk)
            self.knn[gk] = cv.ml.KNearest_create()
            rng = range(self.mrk_ftrs_len,
                        self.mrk_ftrs_len + self.mdl_ftrs_len)
            gX = group[[f'f_{i}' for i in rng]].values.astype(np.float32)
            gy_models = group['ID_model'].values.astype(np.float32)
            self.knn[gk].train(gX, cv.ml.ROW_SAMPLE, gy_models)


class MarksMah:
    def __init__(self, ftrs_base: FeaturesBase, lim: int = 1000,
                 keys: List[str] = None):
        self.df_all = pd.DataFrame(ftrs_base.data_all)
        self.keys = [] if keys is None else keys
        self.ftrs_len = 128
        self.lim = lim  # max amount of points from class to take into account
        self.mah = {}  # {mark_id[, *keys]: (centroid, cov_inv)}

    def forward(self, x, k_neighs=15, keys: dict = None):
        marks_res = []
        dists_res = []
        if not self.keys:
            for i in range(x.shape[0]):
                best_mark = 1
                best_dist = 1e10
                for mrk_id, (centr, cov_inv) in self.mah.items():
                    dist = mahalanobis(x[i:i + 1, :self.ftrs_len], centr, cov_inv)
                    if dist < best_dist:
                        best_mark = mrk_id
                        best_dist = dist
                marks_res.append(best_mark)
                dists_res.append(best_dist)
        else:
            for i in range(x.shape[0]):
                best_mark = 1
                best_dist = 1e10
                for ks, (centr, cov_inv) in self.mah.items():
                    mrk_id = ks[0]
                    t = [keys[kn][i] == ksj for kn, ksj in zip(keys, ks[1:])]
                    if sum(t):
                        dist = mahalanobis(x[i:i + 1, :self.ftrs_len], centr, cov_inv)
                        if dist < best_dist:
                            best_mark = mrk_id
                            best_dist = dist
                    else:
                        continue
                marks_res.append(best_mark)
                dists_res.append(best_dist)
        return np.array(marks_res), np.array(dists_res)

    def set_mah(self):
        grouped = self.df_all.groupby(['ID_mark'] + self.keys)
        for gk in grouped.groups:
            group = grouped.get_group(gk)
            gX = group[[f'f_{i}' for i in range(self.ftrs_len)]].values
            indxs = sample(range(gX.shape[0]), min(self.lim, gX.shape[0]))
            cov_inv = inv_conv_matr(gX[indxs])
            self.mah[gk] = (gX[indxs], cov_inv)


class ModelsMah:
    def __init__(self, ftrs_base: FeaturesBase, lim: int = 1000,
                 keys: List[str] = None):
        self.df_all = pd.DataFrame(ftrs_base.data_all)
        self.keys = [] if keys is None else keys
        self.mrk_ftrs_len = 128
        self.mdl_ftrs_len = 64
        self.lim = lim  # max amount of points from class to take into account
        self.mah = {}  # {model_id: (mark_id, centroid, cov_inv)}

    def forward(self, x, marks_res, k_neighs=15, keys: dict = None):
        models_res = []
        dists_res = []
        if not self.keys:
            for i in range(x.shape[0]):
                best_model = None
                best_dist = 1e10
                for mdl_id, (mrk_id, centr, cov_inv) in self.mah.items():
                    if marks_res[i] == mrk_id:
                        dist = mahalanobis(x[i:i+1, -self.mdl_ftrs_len:],
                                           centr, cov_inv)
                        if dist < best_dist:
                            best_model = mdl_id[0]
                            best_dist = dist
                models_res.append(best_model)
                dists_res.append(best_dist)
        else:
            for i in range(x.shape[0]):
                best_model = 1
                best_dist = 1e10
                for ks, (mrk_id, centr, cov_inv) in self.mah.items():
                    if marks_res[i] == mrk_id:
                        mdl_id = ks[0]
                        t = [keys[kn][i] == ksj for kn, ksj in zip(keys, ks[1:])]
                        if sum(t):
                            dist = mahalanobis(x[i:i + 1, -self.mdl_ftrs_len:],
                                               centr, cov_inv)
                            if dist < best_dist:
                                best_model = mdl_id
                                best_dist = dist
                        else:
                            continue
                models_res.append(best_model)
                dists_res.append(best_dist)
        return np.array(models_res), np.array(dists_res)

    def set_mah(self):
        grouped = self.df_all.groupby(['ID_mark', 'ID_model'] + self.keys)
        for gk in grouped.groups:
            group = grouped.get_group(gk)
            rng = range(self.mrk_ftrs_len,
                        self.mrk_ftrs_len + self.mdl_ftrs_len)
            gX = group[[f'f_{i}' for i in rng]].values
            indxs = sample(range(gX.shape[0]), min(self.lim, gX.shape[0]))
            if gX[indxs].shape[0] < 2:
                continue
            cov_inv = inv_conv_matr(gX[indxs])
            self.mah[tuple([gk[1]] + self.keys)] = (gk[0], gX[indxs], cov_inv)


class MetricClassificationModel:
    def __init__(self, marks_classifier, models_classifier):
        self.marks_classifier = marks_classifier
        self.models_classifier = models_classifier

    def forward(
            self, x: np.array,
            mrks_k_neighs: int = 3,
            mdls_k_neighs: int = 3,
            keys: dict = None
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        mrk_res, mrk_dist = self.marks_classifier.forward(x, mrks_k_neighs,
                                                          keys=keys)
        mdl_res, mdl_dist = self.models_classifier.forward(x, mrk_res,
                                                           mdls_k_neighs,
                                                           keys=keys)
        return mrk_res, mrk_dist, mdl_res, mdl_dist
