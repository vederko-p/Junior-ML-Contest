
from typing import Callable
import IPython.display as IPy_disp

import torch
from torch.utils.data import DataLoader

from models.utils.base_model import BaseModel
from torch.nn import Module as nnModule
from torch.nn import LeakyReLU
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Flatten
from torch.nn import Softmax
from torch.nn import BatchNorm1d
from torch.nn import Dropout


class ConvBlock(nnModule):
    def __init__(self):
        super(ConvBlock, self).__init__()
        # Архитектура:
        _layer_activation = LeakyReLU(0.05)
        self.add_module('activation_LeakyReLU', _layer_activation)

        _conv_0 = Conv2d(1, 4, kernel_size=(5, 5), stride=(4, 4),
                         padding=(2, 2), padding_mode='zeros', bias=True)
        self.add_module('conv00', _conv_0)

        _conv_1 = Conv2d(4, 16, kernel_size=(5, 5), stride=(4, 4),
                         padding=(2, 2), padding_mode='zeros', bias=True)
        self.add_module('conv01', _conv_1)

        _conv_2 = Conv2d(16, 20, kernel_size=(5, 5), stride=(4, 4),
                         padding=(2, 2), padding_mode='zeros', bias=True)
        self.add_module('conv02', _conv_2)

        _conv_3 = Conv2d(20, 22, kernel_size=(3, 3), stride=(1, 1),
                         padding=(1, 1), padding_mode='zeros', bias=True)
        self.add_module('conv03', _conv_3)

        _layer_pooling = MaxPool2d(kernel_size=(2, 2))
        self.add_module('pool_00', _layer_pooling)

        self.add_module('fltn', Flatten())

    def forward(self, x):
        im_01_dwnsmpl = self.conv00(x)
        im_01_dwnsmpl = self.activation_LeakyReLU(im_01_dwnsmpl)
        im_02_dwnsmpl = self.conv01(im_01_dwnsmpl)
        im_02_dwnsmpl = self.activation_LeakyReLU(im_02_dwnsmpl)
        im_03_dwnsmpl = self.conv02(im_02_dwnsmpl)
        im_03_dwnsmpl = self.activation_LeakyReLU(im_03_dwnsmpl)
        im_04_dwnsmpl = self.conv02(im_03_dwnsmpl)
        im_04_dwnsmpl = self.pool_00(im_04_dwnsmpl)
        im_04_dwnsmpl = self.activation_LeakyReLU(im_04_dwnsmpl)
        vect = self.fltn(im_04_dwnsmpl)
        return vect
