
import numpy as np
from scipy.spatial.distance import cdist


def diffusion_metric(a, b, sigma):
    return np.exp(-(np.linalg.norm(a - b)**2 / sigma**2))


def laplas(w_matrix):
    d = np.diag(w_matrix.sum(axis=1))
    return w_matrix - d


class DiffusionMap:
    def __init__(self, kernel):
        self.kernel = kernel

        self.base_data = None
        self.base_labels = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, x, y):
        w_matrix = cdist(x, x, metric=self.kernel)
        laplasian = laplas(w_matrix)
        w, v = np.linalg.eig(laplasian)  # w[i] - eigenval, v[:, i] - eigenvect
        order = np.argsort(abs(w))
        self.base_data = x.clone().numpy()
        self.base_labels = y.clone().numpy()
        self.eigenvalues = w[order]
        self.eigenvectors = v[:, order]

    def transform_base(self, ids, k=3):
        q = min(k, self.eigenvectors.shape[1])
        return self.eigenvectors[ids, -q:], self.base_labels[ids]


if __name__ == '__main__':
    SIGMA = 300

    def dm_kernel(x, y):
        return diffusion_metric(x, y, SIGMA)

    diffusion_map = DiffusionMap(dm_kernel)

    vectors = 'vectors'
    labels = 'labels'
    diffusion_map.fit(vectors, labels)
