
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
        
        _conv_2 = Conv2d(16, 22, kernel_size=(3, 3), stride=(1, 1),
                         padding=(1, 1), padding_mode='zeros', bias=True)        
        self.add_module('conv02', _conv_2) 
        
        _layer_pooling = MaxPool2d(kernel_size=(2, 2))
        self.add_module('pool_00', _layer_pooling)
        
        self.add_module('fltn', Flatten())
        
    def forward(self, x):
        im_01_dwnsmpl = self.conv00(x)
        im_01_dwnsmpl = self.activation_LeakyReLU(im_01_dwnsmpl)
        im_02_dwnsmpl = self.conv01(im_01_dwnsmpl)
        im_02_dwnsmpl = self.activation_LeakyReLU(im_02_dwnsmpl)
        im_03_dwnsmpl = self.conv02(im_02_dwnsmpl)
        im_03_dwnsmpl = self.pool_00(im_03_dwnsmpl)
        im_03_dwnsmpl = self.activation_LeakyReLU(im_03_dwnsmpl)
        vect = self.fltn(im_03_dwnsmpl)
        return vect
        

class FullyConnectClassifyBlock(nnModule):
    def __init__(self, numclasses):
        super(FullyConnectClassifyBlock, self).__init__()
        # Параметры:
        self.numclasses = numclasses
        # Архитектура:
        _layer_activation = LeakyReLU(0.05)
        self.add_module('activation_LeakyReLU', _layer_activation)
        
        _linear_0 = Linear(352, 256, bias=True)
        self.add_module('linear00', _linear_0)
        _dropout_0 = Dropout(0.5)
        self.add_module('dropout00', _dropout_0)
        
        _batch_norm_0 = BatchNorm1d(256)
        self.add_module('batch_norm00',  _batch_norm_0)
        _linear_1 = Linear(256, 128, bias=True)
        self.add_module('linear01', _linear_1)
        
        _linear_2 = Linear(128,  self.numclasses, bias=True)
        self.add_module('linear02', _linear_2)
       
        _SftMax = Softmax(dim=-1)
        self.add_module('SftMax', _SftMax)

    def forward(self, x):
        vect_01 = self.linear00(x)
        vect_01 = self.dropout00(vect_01)
        vect_01 = self.activation_LeakyReLU(vect_01)
        vect_02 = self.batch_norm00(vect_01)
        vect_02 = self.linear01(vect_02)
        vect_02 = self.activation_LeakyReLU(vect_02)
        vect_03 = self.linear02(vect_02)
        vect_03 = self.SftMax(vect_03)
        return vect_03
    
    
class FullyConnectTLBlock(nnModule):
    def __init__(self, n_out):
        super(FullyConnectTLBlock, self).__init__()
        # Параметры:
        self.n_out = n_out
        # Архитектура:
        _layer_activation = LeakyReLU(0.05)
        self.add_module('activation_LeakyReLU', _layer_activation)
        
        _linear_0 = Linear(352, 256, bias=True)
        self.add_module('linear00', _linear_0)
        _dropout_0 = Dropout(0.5)
        self.add_module('dropout00', _dropout_0)
        
        _batch_norm_0 = BatchNorm1d(256)
        self.add_module('batch_norm00',  _batch_norm_0)
        _linear_1 = Linear(256, 128, bias=True)
        self.add_module('linear01', _linear_1)
        
        _linear_2 = Linear(128,  self.n_out, bias=True)
        self.add_module('linear02', _linear_2)

    def forward(self, x):
        vect_01 = self.linear00(x)
        vect_01 = self.dropout00(vect_01)
        vect_01 = self.activation_LeakyReLU(vect_01)
        vect_02 = self.batch_norm00(vect_01)
        vect_02 = self.linear01(vect_02)
        vect_02 = self.activation_LeakyReLU(vect_02)
        vect_03 = self.linear02(vect_02)
        return vect_03


class TripletLossModel(BaseModel):
    def __init__(self, n_out):
        super(TripletLossModel, self).__init__()
        # Параметры:
        self.n_out = n_out
        # Архитектура:
        self.conv2Dfeatures = ConvBlock()
        self.fully_connect = FullyConnectTLBlock(self.n_out)
        
    def forward(self, x):
        vect00 = self.conv2Dfeatures.forward(x)
        vect01 = self.fully_connect.forward(vect00)
        return vect01

    def fit(self, dataset, epochs, batch_size=32,
            callback: Callable[[nnModule, float], None] = None) -> None:
        dataloader = DataLoader(dataset, batch_size=batch_size)
        last_loss = 'none'
        for ep in range(epochs):
            IPy_disp.clear_output(True)
            print(f'epoch: {ep+1}/{epochs} | loss: {last_loss}\n')
            epoch_loss = self.train_epoch(dataloader)
            last_loss = round(epoch_loss, 3)
            if callback is not None:
                callback(self, epoch_loss)
    
    def train_epoch(self, dataloader):
        self.train()
        epoch_loss = 0
        for it, batch in enumerate(dataloader):
            batch_loss = self.train_on_batch(batch)
            epoch_loss += batch_loss
            print('.', end='')
        print('')
        return epoch_loss / (it+1)
    
    def train_on_batch(self, batch):
        data_anc, data_pos, data_neg = self.prepare_data(batch)
        out_anc = self.forward(data_anc)
        out_pos = self.forward(data_pos)
        out_neg = self.forward(data_neg)
        loss = self.criterion(out_anc, out_pos, out_neg)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.cpu().item()
    
    def prepare_data(self, batch):
        data_anc = batch['Anchor']
        data_anc.to(self.device)
        data_pos = batch['Positive']
        data_pos.to(self.device)
        data_neg = batch['Negative']
        data_neg.to(self.device)
        return data_anc, data_pos, data_neg
    
    def save_state(self, file_name):
        torch.save(self.state_dict(), file_name)
        return 0

    def split_save(self, file_names):
        torch.save(self.conv2Dfeatures.state_dict(), file_names[0])
        torch.save(self.fully_connect.state_dict(), file_names[1])

    def load_state(self, file_name):
        self.load_state_dict(torch.load(file_name))
        return 0

    def split_load(self, file_names, map_location=None):
        self.conv2Dfeatures.load_state_dict(
            torch.load(file_names[0], map_location))
        self.fully_connect.load_state_dict(
            torch.load(file_names[1], map_location))
        return

    def load_conv_block(self, filepath, map_location=None):
        self.conv2Dfeatures.load_state_dict(torch.load(filepath, map_location))

    def load_fc_block(self, filepath, map_location=None):
        self.fully_connect.load_state_dict(torch.load(filepath, map_location))

    def load_warmstart_fc_block(self, filepath, map_location=None):
        weights = torch.load(filepath, map_location)
        del weights['linear02.weight']
        del weights['linear02.bias']
        _ = self.fully_connect.load_state_dict(weights, strict=False)


class ClassificationForTLModel(BaseModel):
    def __init__(self, numclasses):
        super(ClassificationForTLModel, self).__init__()
        # Параметры:
        self.numclasses = numclasses
        # Архитектура:
        self.conv2Dfeatures = ConvBlock()
        self.fully_connect = FullyConnectClassifyBlock(self.numclasses)

    def forward(self, x):
        vect00 = self.conv2Dfeatures.forward(x)
        vect01 = self.fully_connect.forward(vect00)
        return vect01

    def fit(self, dataset, epochs, batch_size=32,
            callback: Callable[[nnModule, float], None] = None) -> None:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, drop_last=True)
        last_loss = 'none'
        for ep in range(epochs):
            IPy_disp.clear_output(True)
            print(f'epoch: {ep+1}/{epochs} | loss: {last_loss}\n')
            epoch_loss = self.train_epoch(dataloader)
            last_loss = round(epoch_loss, 3)
            if callback is not None:
                callback(self, epoch_loss)
    
    def train_epoch(self, dataloader):
        self.train()
        epoch_loss = 0
        for it, batch in enumerate(dataloader):
            batch_loss = self.train_on_batch(batch)
            epoch_loss += batch_loss
            print('.', end='')
        print('')
        return epoch_loss / (it+1)
    
    def train_on_batch(self, batch):
        data, classes = self.prepare_data(batch)
        out = self.forward(data)
        loss = self.criterion(out, classes)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.cpu().item()
    
    def prepare_data(self, batch):
        data = batch[0].to(self.device)
        classes = batch[1].to(self.device)
        return data, classes
    
    def save_state(self, file_name):
        torch.save(self.state_dict(), file_name)
        return
    
    def split_save(self, file_names):
        torch.save(self.conv2Dfeatures.state_dict(), file_names[0])
        torch.save(self.fully_connect.state_dict(), file_names[1])
    
    def load_state(self, file_name):
        self.load_state_dict(torch.load(file_name))
        return
    
    def split_load(self, file_names, map_location=None):
        self.conv2Dfeatures.load_state_dict(
            torch.load(file_names[0], map_location))
        self.fully_connect.load_state_dict(
            torch.load(file_names[1], map_location))
        return
