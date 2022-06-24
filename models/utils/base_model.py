
from torch.nn import Module as nnModule
import torch


class BaseModel(nnModule):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.optimizer = None
        self.criterion = None
        self.device = None

    def compile_settings(self, criterion, optimizer, device=None):
        self.optimizer = optimizer
        self.criterion = criterion
        if device is None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        self.to(self.device)
