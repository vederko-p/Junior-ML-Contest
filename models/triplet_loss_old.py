
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import Conv2d
from torch.nn import LeakyReLU
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import Dropout
from torch.nn import Sigmoid
from torch.nn import BatchNorm1d
from torch.nn import Flatten

from models.utils.regularizer import Regularizer


class Layer_06(torch.nn.Module):
    def __init__(self, *input_shapes, **kwargs):
        super(Layer_06, self).__init__(**kwargs)
        self.input_shapes = input_shapes
        self.eps_ = 10**(-20)
        self._criterion = None
        self._optimizer = None

    def reset_parameters(self):
        def hidden_init(layer):
            fan_in = layer.weight.data.size()[0]
            lim = 1. / np.sqrt(fan_in)
            return (-lim, lim)

        for module in self._modules.values():
            if hasattr(module, 'weight') and (module.weight is not None):
                module.weight.data.uniform_(*hidden_init(module))
            if hasattr(module, 'bias') and (module.bias is not None):
                module.bias.data.fill_(0)

    def _get_regularizer(self):
        raise Exception("Need to override method _get_regularizer()!")

    def weights_is_nan(self):
        is_nan = False
        for module in self._modules.values():
            if hasattr(module, 'weight'):
                if ((isinstance(module.weight, torch.Tensor) and torch.isnan
                        (torch.sum(module.weight.data).detach())) or
                        (isinstance(module.weight, torch.Tensor) and torch.isnan
                            (torch.sum(module.weight.data).detach()))):
                    is_nan = True
                    break
            if hasattr(module, 'bias'):
                if (isinstance(module.bias, torch.Tensor) and torch.isnan
                        (torch.sum(module.bias.data).detach())):
                    is_nan = True
                    break

        return is_nan

    def save_state(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_state(self, file_path, map_location = None):
        try:
            print()
            print('Loading preset weights... ', end='')

            self.load_state_dict(torch.load(file_path, map_location))
            self.eval()
            is_nan = False
            for module in self._modules.values():
                if hasattr(module, 'weight'):
                    if ((isinstance(module.weight, torch.Tensor) and torch.isnan
                            (torch.sum(module.weight.data).detach())) or
                            (isinstance(module.weight, torch.Tensor) and torch.isnan
                                (torch.sum(module.weight.data).detach()))):
                        is_nan = True
                        break
                if hasattr(module, 'bias'):
                    if (isinstance(module.bias, torch.Tensor) and torch.isnan
                            (torch.sum(module.bias.data).detach())):
                        is_nan = True
                        break

            if (is_nan):
                raise Exception("[Error]: Parameters of layers is NAN!")

            print("Ok.")
        except Exception as e:
            print("Fail! ", end='')
            print(str(e))
            print("[Action]: Reseting to random values!")
            self.reset_parameters()

    def cross_entropy_00(self, pred, soft_targets):
        return -torch.log(self.eps_
                          + torch.mean(torch.sum(soft_targets * pred, -1)))

    def MSE_00(self, pred, soft_targets):
        return torch.mean(torch.mean((soft_targets - pred)**2, -1))

    def compile(self, criterion, optimizer, **kwargs):

        if criterion == 'mse-mean':
            self._criterion = nn.MSELoss(reduction='mean')
        elif criterion == 'mse-sum':
            self._criterion = nn.MSELoss(reduction='sum')
        elif criterion == '000':
            self._criterion = self.MSE_00
        elif criterion == '001':
            self._criterion = self.cross_entropy_00

        else:
            raise Exception("Unknown loss-function!")

        if optimizer == 'sgd':
            momentum = 0.2
            if 'lr' in kwargs.keys():
                lr = kwargs['lr']
            if 'momentum' in kwargs.keys():
                momentum = kwargs['momentum']
            self._optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        elif optimizer == 'adam':
            if 'lr' in kwargs.keys():
                lr = kwargs['lr']
            self._optimizer   = torch.optim.Adam(self.parameters(), lr=lr,
                                                 betas=(0.9, 0.999),
                                                 eps=1e-08, weight_decay=0,
                                                 amsgrad=False)
        else:
            raise Exception("Unknown optimizer!")

    def _call_simple_layer(self, name_layer, x):
        y = self._modules[name_layer](x)
        if self.device.type == 'cuda' and not y.is_contiguous():
            y = y.contiguous()
        return y

    def _contiguous(self, x):
        if self.device.type == 'cuda' and not x.is_contiguous():
            x = x.contiguous()
        return x


class conv_simple_features_01(Layer_06):
    def __init__(self, device=None, L1=0., L2=0., show=0):
        super(conv_simple_features_01, self).__init__()
        self.show = show
        if device is not None:
            self.device = device if (
                not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.L1 = L1
        self.L2 = L2
        self.regularizer = Regularizer(L1, L2)
        _layer_conv_31 = Conv2d(1, 4, kernel_size=(5, 5),
                                stride=(4, 4), padding=(2, 2),
                                padding_mode='zeros', bias=True)
        self.add_module('conv00', _layer_conv_31)
        _layer_activation_1 = LeakyReLU(0.05)
        self.add_module('activation_LeakyReLU', _layer_activation_1)
        _layer_conv_32 = Conv2d(4, 16, kernel_size=(5, 5),
                                stride=(4, 4), padding=(2, 2),
                                padding_mode='zeros', bias=True)
        self.add_module('conv01', _layer_conv_32)
        _layer_conv_33 = Conv2d(16, 20, kernel_size=(3, 3),
                                stride=(1, 1), padding=(1, 1),
                                padding_mode='zeros', bias=True)

        self.add_module('conv02', _layer_conv_33)
        _layer_conv_34 = Conv2d(20, 22, kernel_size=(3, 3),
                                stride=(1, 1), padding=(1, 1),
                                padding_mode='zeros', bias=True)

        self.add_module('conv03', _layer_conv_34)
        _layer_pooling_1 = MaxPool2d(kernel_size=(2, 2))
        self.add_module('Pool_00', _layer_pooling_1)
        self.add_module('fltn_1', Flatten())

        self.to(self.device)
        self.reset_parameters()

    def forward(self, scatch0):
        im_01_dwnsmpl = self.conv00(scatch0)
        im_01_dwnsmpl = self.activation_LeakyReLU(im_01_dwnsmpl)
        if self.show:
            print('im_01_dwnsmpl', im_01_dwnsmpl.shape)
        im_02_dwnsmpl = self.conv01(im_01_dwnsmpl)
        im_02_dwnsmpl = self.activation_LeakyReLU(im_02_dwnsmpl)
        if self.show:
            print('im_02_dwnsmpl', im_02_dwnsmpl.shape)

        im_02a_dwnsmpl = self.conv02(im_02_dwnsmpl)
        im_02a_dwnsmpl = self.activation_LeakyReLU(im_02a_dwnsmpl)
        if self.show:
            print('im_02a_dwnsmpl', im_02a_dwnsmpl.shape)

        im_03_dwnsmpl = self.conv03(im_02a_dwnsmpl)
        im_03_dwnsmpl = self.Pool_00(im_03_dwnsmpl)
        im_03_dwnsmpl = self.activation_LeakyReLU(im_03_dwnsmpl)
        if self.show:
            print('im_03_dwnsmpl', im_03_dwnsmpl.shape)
        vect_00 = self.fltn_1(im_03_dwnsmpl)
        if self.show:
            print('vect_00', vect_00.shape)
        return vect_00


class fully_connect_modul_266(Layer_06):
    def __init__(self, device=None, L1=0., L2=0., show=0):
        super(fully_connect_modul_266, self).__init__()
        self.show = show
        if device is not None:
            self.device = device if (
                not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.L1 = L1
        self.L2 = L2
        self.regularizer = Regularizer(L1, L2)
        _layer_activation_1 = LeakyReLU(0.05)
        self.add_module('activation_LeakyReLU', _layer_activation_1)
        _layer_D01 = Linear(352, 300, bias=True)
        self.add_module('D01', _layer_D01)
        _layer_Dropout01 = Dropout(0.1)
        _layer_batch_norm_3 = BatchNorm1d(256)
        self.add_module('Dropout01', _layer_Dropout01)
        self.add_module('layer_batch_norm', _layer_batch_norm_3)

        _layer_D02 = Linear(300, 256, bias=True)
        self.add_module('D02', _layer_D02)
        _layer_Dropout02 = Dropout(0.1)

        self.add_module('Dropout02', _layer_Dropout01)

        _layer_D03 = Linear(256, 200, bias=True)
        self.add_module('D03', _layer_D03)
        _layer_D04 = Linear(200, 128, bias=True)
        self.add_module('D04', _layer_D04)

        _layer_Sgmd = Sigmoid()
        self.add_module('Sgmd', _layer_Sgmd)

        self.to(self.device)
        self.reset_parameters()

    def forward(self, vect_00):
        vect_01 = self.D01(vect_00)
        vect_01 = self.Dropout01(vect_01)
        vect_01 = self.activation_LeakyReLU(vect_01)
        if self.show:
            print('vect_01', vect_01.shape)

        vect_02 = self.D02(vect_01)
        vect_02 = self.Dropout02(vect_02)
        vect_02 = self.activation_LeakyReLU(vect_02)
        if self.show:
            print('vect_02', vect_02.shape)
        vect_03 = self.D03(vect_02)
        vect_03 = self.activation_LeakyReLU(vect_03)
        vect_04 = self.D04(vect_03)
        if self.show:
            print('vect_03', vect_03.shape)
            print('vect_04', vect_04.shape)
        return vect_04


class conv_simple_features_00(Layer_06):
    def __init__(self, device=None, L1=0., L2=0., show=0):
        super(conv_simple_features_00, self).__init__()
        self.show = show
        if (device is not None):
            self.device = device if (
                not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.L1 = L1
        self.L2 = L2
        self.regularizer = Regularizer(L1, L2)
        _layer_conv_31 = Conv2d(1, 4, kernel_size=(5, 5),
                                stride=(4, 4), padding=(2, 2),
                                padding_mode='zeros', bias=True)
        self.add_module('conv00', _layer_conv_31)
        _layer_activation_1 = LeakyReLU(0.05)
        self.add_module('activation_LeakyReLU', _layer_activation_1)
        _layer_conv_32 = Conv2d(4, 16, kernel_size=(5, 5),
                                stride=(4, 4), padding=(2, 2),
                                padding_mode='zeros', bias=True)

        self.add_module('conv01', _layer_conv_32)
        _layer_conv_33 = Conv2d(16, 22, kernel_size=(3, 3),
                                stride=(1, 1), padding=(1, 1),
                                padding_mode='zeros', bias=True)

        self.add_module('conv02', _layer_conv_33)
        _layer_pooling_1 = MaxPool2d(kernel_size=(2, 2))
        self.add_module('Pool_00', _layer_pooling_1)
        self.add_module('fltn_1', Flatten())

        self.to(self.device)
        self.reset_parameters()

    def forward(self, scatch0):
        im_01_dwnsmpl = self.conv00(scatch0)
        im_01_dwnsmpl = self.activation_LeakyReLU(im_01_dwnsmpl)
        if self.show:
            print('im_01_dwnsmpl', im_01_dwnsmpl.shape)
        im_02_dwnsmpl = self.conv01(im_01_dwnsmpl)
        im_02_dwnsmpl = self.activation_LeakyReLU(im_02_dwnsmpl)
        if self.show:
            print('im_02_dwnsmpl', im_02_dwnsmpl.shape)
        im_03_dwnsmpl = self.conv02(im_02_dwnsmpl)
        im_03_dwnsmpl = self.Pool_00(im_03_dwnsmpl)
        im_03_dwnsmpl = self.activation_LeakyReLU(im_03_dwnsmpl)
        if self.show:
            print('im_03_dwnsmpl', im_03_dwnsmpl.shape)
        vect_00 = self.fltn_1(im_03_dwnsmpl)
        if self.show:
            print('vect_00', vect_00.shape)
        return vect_00


class fully_connect_modul_265(Layer_06):
    def __init__(self, device=None, L1=0., L2=0., show=0):
        super(fully_connect_modul_265, self).__init__()
        self.show = show
        if device is not None:
            self.device = device if (
                not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.L1 = L1
        self.L2 = L2
        self.regularizer = Regularizer(L1, L2)
        _layer_activation_1 = LeakyReLU(0.05)
        self.add_module('activation_LeakyReLU', _layer_activation_1)
        _layer_D01 = Linear(352, 256, bias=True)
        self.add_module('D01', _layer_D01)
        _layer_Dropout01 = Dropout(0.5)
        _layer_batch_norm_3 = BatchNorm1d(256)
        self.add_module('Dropout01', _layer_Dropout01)
        self.add_module('layer_batch_norm', _layer_batch_norm_3)
        _layer_D02 = Linear(256, 128, bias=True)
        self.add_module('D02', _layer_D02)
        _layer_D03 = Linear(128, 64, bias=True)
        self.add_module('D03', _layer_D03)

        _layer_Sgmd = Sigmoid()
        self.add_module('Sgmd', _layer_Sgmd)
        #########################
        self.to(self.device)
        self.reset_parameters()

    def forward(self, vect_00):
        vect_01 = self.D01(vect_00)
        vect_01 = self.Dropout01(vect_01)
        vect_01 = self.activation_LeakyReLU(vect_01)
        if self.show:
            print('vect_01', vect_01.shape)
        vect_01 = self.layer_batch_norm(vect_01)
        if self.show:
            print('vect_01 layer_batch_norm', vect_01.shape)
        vect_02 = self.D02(vect_01)
        vect_02 = self.activation_LeakyReLU(vect_02)
        if self.show:
            print('vect_02', vect_02.shape)
        vect_03 = self.D03(vect_02)

        if self.show:
            print('vect_03', vect_03.shape)
        return vect_03


class TL_005_concatenate(Layer_06):
    def __init__(self, imageSize, L1=0., L2=0., device=None, show=0, n_out=192):
        super(TL_005_concatenate, self).__init__(
            (imageSize[0], imageSize[1], 1))

        self.class_name = self.__class__.__name__
        self.n_out = n_out

        self.imageSize = imageSize
        self.regularizer = Regularizer(L1, L2)
        self.show = show
        self.L1 = L1
        self.L2 = L2

        if device is not None:
            self.device = device if (
                not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.conv_newModel = conv_simple_features_01(device, L1, L2, show)
        self.FC_TL_newModel = fully_connect_modul_266(device, L1, L2, show)
        self.conv_oldModel = conv_simple_features_00(device, L1, L2, show)
        self.FC_TL_oldModel = fully_connect_modul_265(device, L1, L2, show)

        self.to(self.device)
        self.reset_parameters()

    def forward(self, x):
        vector_new_00 = self.conv_newModel(x)
        vector_new_01 = self.FC_TL_newModel(vector_new_00)

        vector_old_00 = self.conv_oldModel(x)
        vector_old_01 = self.FC_TL_oldModel(vector_old_00)

        vector_res = torch.cat((vector_new_01, vector_old_01), axis=1)
        x = self._contiguous(vector_res)

        return x.squeeze()


class TripletLossModelOld(Layer_06):
    def __init__(self, conv_block, fc_block, imageSize,
                 L1=0., L2=0., device=None, show=0, n_out=64):
        super(TripletLossModelOld, self).__init__(
            (imageSize[0], imageSize[1], 1))

        self.class_name = self.__class__.__name__
        self.n_out = n_out

        self.imageSize = imageSize
        self.regularizer = Regularizer(L1, L2)
        self.show = show
        self.L1 = L1
        self.L2 = L2

        if device is not None:
            self.device = device if (
                not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.conv_block = conv_block(device, L1, L2, show)
        self.fc_block = fc_block(device, L1, L2, show)

        self.to(self.device)
        self.reset_parameters()

    def forward(self, x):
        vector_00 = self.conv_block(x)
        vector_01 = self.fc_block(vector_00)
        return vector_01
