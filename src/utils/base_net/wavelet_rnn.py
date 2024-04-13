import torch
from torch import nn
import numpy as np


class WaveletRNNCell(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, device):
        super(WaveletRNNCell, self).__init__()
        total_dim = inp_dim + out_dim + hidden_dim
        hidden_dim = hidden_dim
        self._device = device
        
        self.h_func_1 = nn.Linear(total_dim, total_dim).to(self._device)
        self.h_func_2_list = []
        for i in range(1,hidden_dim):
            h_func = nn.Linear(total_dim, total_dim).to(self._device)
            self.h_func_2_list.append(h_func)
        
        self.y_func = nn.Linear(total_dim, out_dim).to(self._device)
                
    def activate_func_1(self, x):
        return torch.prod(torch.exp(-0.5 * torch.pow(x, 2))).unsqueeze(0)
        
    def activate_func_2(self, x):
        return torch.prod((1 - torch.pow(x, 2)) * torch.exp(-0.5 * torch.pow(x, 2))).unsqueeze(0)

    def forward(self, h, y, u):
        # print('h shape:{}'.format(h.shape))
        # print('y shape:{}'.format(y.shape))
        # print('u shape:{}'.format(u.shape))
        input = torch.cat((h,y,u),dim=1)
        # print('input shape:{}'.format(input.shape))
        y_output = self.y_func(input)
        h_output = []
        h11 = self.h_func_1(input)
        h_1_output = self.activate_func_1(h11)
        h_output.append(h_1_output)
        for h_func_2 in self.h_func_2_list:
            # print('net device:{}'.format(h_func_2.device))
            # print('input device:{}'.format(input.device))
            h_i = h_func_2(input)
            h_a = self.activate_func_2(h_i)
            h_output.append(h_a)
        h_output = torch.Tensor(h_output).to(self._device)
        return h_output[np.newaxis, : ], y_output
        
class WaveletRNN(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, device):
        super(WaveletRNN, self).__init__()
        self.wavelet_rnn_cell = WaveletRNNCell(inp_dim, out_dim, hidden_dim, device)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device

    def forward(self, x, hy_0=None):
        batch_size, seq_len, input_size = x.size()
        
        if hy_0 is not None:
            h_0, y_0 = hy_0
        else:
            h_0 = torch.zeros(batch_size, self.hidden_dim).to(self.device)
            y_0 = torch.zeros(batch_size, self.out_dim).to(self.device)
            
        last_h = h_0
        last_y = y_0
        h_list = []
        
        for t in range(seq_len):
            #x_t.shape = (batch_size ,1, input_size)
            x_t = x[:,t,:]
            h_output, y_output = self.wavelet_rnn_cell(last_h, last_y, x_t)
            h_list.append(h_output)
            last_h, last_y = h_output, y_output
        
        # h_list = [(batch_size,hidden_size) * seq_len]
        # [(batch_size,1,hidden_size)]
        # h.shape=(batch_size,seq_len,hidden_size)
        
        h = torch.stack(h_list, dim = 1).to(self.device)
        return y_output