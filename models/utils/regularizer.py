
import torch


class Regularizer:
    """Regularizer for L1 and L2 regularization.

    Parameters
    ----------
    l1 : `float`
        L1 regularization factor.
    l2 : `float`
        L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = float(l1)  # torch.FloatTensor(l1)
        self.l2 = float(l2)  # torch.FloatTensor(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += self.l1 * torch.sum(torch.abs(x))
        if self.l2:
            regularization += self.l2 * torch.sqrt(torch.sum(torch.pow(x, 2)))
        return regularization
