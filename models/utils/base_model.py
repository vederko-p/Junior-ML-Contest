
from torch.nn import Module as nnModule
import torch


class BaseModel(nnModule):
    def __init__(self):
        super(BaseModel, self).__init__()

    def compile_settings(self, criterion, optimizer, device=None, log_module=None):
        self.optimizer = optimizer
        self.criterion = criterion
        self.loger = log_module  # loger не должен быть в compile, он должен передаваться через call back
        if device is None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        self.to(self.device)
