import torch
from torch import nn
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size, :-self.chomp_size].contiguous()
    
class FCN_unit(nn.Module):
    def __init__(self, index, input_channel, output_channel, kernel_size, active_func, dropout):
        super(FCN_unit, self).__init__()
        self._unit = nn.Sequential()
        stride = 1
        padding = 1
        dilation = 1
        self._unit.add_module("Conv_{}".format(index), nn.Conv2d(input_channel, 
                                                                 output_channel, 
                                                                 kernel_size, 
                                                                 stride=stride, 
                                                                 padding=padding, 
                                                                 dilation=dilation))
        self._unit.add_module("Chomp_{}".format(index), Chomp1d(padding))
        self._unit.add_module("BN_{}".format(index), nn.BatchNorm2d(output_channel))
        if(active_func=='relu'):
            self._unit.add_module("ReLU_{}".format(index), nn.ReLU())
        elif(active_func=='tanh'):
            self._unit.add_module("Tanh_{}".format(index), nn.Tanh())
        self._unit.add_module("Dropout_{}".format(index), nn.Dropout(dropout))
        
    def forward(self, x):
        return self._unit(x)