
import torch
from torch.nn import MSELoss, TripletMarginLoss


class TMLMSELoss:
    def __init__(self,
                 tml_coef: float = 1.0,
                 tml_margin: float = 30,
                 tml_p: float = 2,
                 mse_coeff: float = 2.0):
        self.tml_loss = TripletMarginLoss(margin=tml_margin, p=tml_p)
        self.mse_loss = MSELoss()
        self.tml_coef = tml_coef
        self.mse_coeff = mse_coeff

    def __call__(self,
                 anc: torch.tensor,
                 pos: torch.tensor,
                 neg: torch.tensor) -> torch.Tensor:
        tml_loss = self.tml_loss(anc, pos, neg)
        mse_loss = self.mse_loss(anc, pos)
        return self.tml_coef*tml_loss + self.mse_coeff*mse_loss


class TMLRegularizedEmbeddingsLoss:
    def __init__(self,
                 tml_coef: float = 1.0,
                 tml_margin: float = 30,
                 tml_p: float = 2,
                 C: float = 0):
        self.tml_loss = TripletMarginLoss(margin=tml_margin, p=tml_p)
        self.tml_coef = tml_coef
        self.C = C

    def __call__(self,
                 anc: torch.tensor,
                 pos: torch.tensor,
                 neg: torch.tensor) -> torch.Tensor:
        tml_loss = self.tml_loss(anc, pos, neg)
        embed_loss_anc = torch.linalg.norm(anc)
        embed_loss_pos = torch.linalg.norm(pos)
        embed_loss_neg = torch.linalg.norm(neg)
        loss = self.tml_coef*tml_loss + self.C*(embed_loss_anc
                                                + embed_loss_pos
                                                + embed_loss_neg)
        return loss
