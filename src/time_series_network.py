#!/usr/bin/env python
import math
import os
import shutil
import datetime
import torch
import yaml
import argparse
import random
import numpy as np
import pandas as pd
import torch.distributed as dist
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from utils.base_net.base_ffnn import BaseFFNN
from utils.base_net.base_lstm import BaseLSTM
from utils.base_net.base_gru import BaseGRU
from utils.base_net.wavelet_rnn import WaveletRNN
from utils.ffnn_data_loader import FFNNDataLoader
from utils.rnn_data_loader import RNNDataLoader
from utils.visualizer import Visualizer
from torch.utils.data import DataLoader
from utils.time_series_dataset import TimeSeriesDataset

seed = 999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class TimeSeriesNetwork:
    def __init__(self, config):
        self._data_loader_config = config['data_loader']
        self._plot_config = config['plot']
        self._train_config = config['train']
        self._type = self._train_config['net_type']
        self._reach_min_loss_flag = False
        self._init_time = str(datetime.datetime.now()).replace(' ','_')
        
        self._initialized_flag = False
        
        self._train_comparison_epoch = []
        self._test_comparison_epoch = []
        self._test_loss_epoch = []
        self._train_loss = []
        
        self._model_dir = os.path.join(root_path, 'model', self._data_loader_config['data_dir'], self._type)
        if(args.task == 'train'):
            self._middle_model_dir = os.path.join(self._model_dir, '{}'.format(self._init_time))
            if os.path.exists(self._middle_model_dir):
                shutil.rmtree(self._middle_model_dir)
            os.makedirs(self._middle_model_dir)
            src_config = os.path.join(root_path,'config',args.config_filepath)
            tgt_config = os.path.join(self._middle_model_dir,args.config_filepath)
            shutil.copyfile(src_config, tgt_config) 
        self._device = torch.device('cuda:{}'.format(self._train_config['cuda_id']) if torch.cuda.is_available() else "cpu")
        self.init_data_loader()
        self._visualizer = Visualizer(self._plot_config)
        
    def init_data_loader(self):
        if self._type == 'LSTM' or self._type == 'GRU' or self._type == 'WRNN' :
            self._data_loader = RNNDataLoader(self._data_loader_config, self._device)
        elif self._type == 'FFNN':
            self._data_loader = FFNNDataLoader(self._data_loader_config, self._device)
        else:
            self._data_loader = RNNDataLoader(self._data_loader_config, self._device)

    def load_all_data(self, data_dir):
        self._data_dict, self._inp_dim = self._data_loader.load_data_dict(data_dir, self._data_loader_config['data_range'])   
        print('train data is loaded successfully!')
               
    def initialize_model(self):
        if not self._initialized_flag:
            if self._type == 'LSTM':
                self._total_epoch = self._train_config['LSTM_args']['total_epoch']
                self._out_dim = self._train_config['LSTM_args']['out_dim']
                self._hidden_dim = self._train_config['LSTM_args']['hidden_dim']
                self._num_layers = self._train_config['LSTM_args']['num_layers']
                self._net = BaseLSTM(self._inp_dim, self._out_dim, self._hidden_dim, self._num_layers).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._train_config['LSTM_args']['learning_rate'])
            elif self._type == 'GRU':
                self._total_epoch = self._train_config['GRU_args']['total_epoch']
                self._out_dim = self._train_config['GRU_args']['out_dim']
                self._hidden_dim = self._train_config['GRU_args']['hidden_dim']
                self._num_layers = self._train_config['GRU_args']['num_layers']
                self._net = BaseGRU(self._inp_dim, self._out_dim, self._hidden_dim, self._num_layers).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._train_config['GRU_args']['learning_rate'])
            elif self._type == 'FFNN':
                self._total_epoch = self._train_config['FFNN_args']['total_epoch']
                self._out_dim = self._train_config['FFNN_args']['output_dim']
                self._net = BaseFFNN(self._inp_dim, self._train_config['FFNN_args']['dim_list'], self._out_dim).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._train_config['FFNN_args']['learning_rate'])
            elif self._type == 'WRNN':
                self._total_epoch = self._train_config['WRNN_args']['total_epoch']
                self._out_dim = self._train_config['WRNN_args']['out_dim']
                self._hidden_dim = self._train_config['WRNN_args']['hidden_dim']
                self._net = WaveletRNN(self._inp_dim, self._out_dim, self._hidden_dim, self._device).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._train_config['WRNN_args']['learning_rate'])
                
            else:
                self._total_epoch = self._train_config['FFNN_args']['total_epoch']
                self._out_dim = self._train_config['FFNN_args']['output_dim']
                self._net = BaseFFNN(self._inp_dim, self._train_config['FFNN_args']['dim_list'], self._out_dim).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._train_config['FFNN_args']['learning_rate']) 
            self._initialized_flag = True
            self._criterion = nn.MSELoss().to(self._device)
            self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, 
                                                              step_size = self._train_config['scheduler_step_size'], 
                                                              gamma = self._train_config['scheduler_gamma'])
        else:
            print('model has been initialized!')
            
        self._start_time = datetime.datetime.now()
            
    def get_total_epoch(self):
        return self._total_epoch
    
    def get_reach_min_loss_flag(self):
        return self._reach_min_loss_flag

    def get_model_dir(self):
        return self._model_dir
    
    def train(self):
        train_data_assembly = []
        for name, train_data in self._data_dict.items():
            train_data_assembly.extend(train_data)
        batch_size = self._data_loader_config['train_batch_num']
        train_dataset = TimeSeriesDataset(train_data_assembly)
        train_data_loader = DataLoader(train_dataset, 
                                       batch_size = batch_size , 
                                       shuffle = self._train_config['shuffled'],
                                       )
        total = int(len(train_dataset)/batch_size)
        start_epoch = 0
        epoch_loss = 0.0
        epoch_loop = tqdm(range(start_epoch, self._total_epoch), 
                          bar_format='{l_bar}|{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining} |{rate_fmt}{postfix}]',
                          leave=True)
        for epoch in epoch_loop:
            epoch_loss_sum = 0.0
            output_merged = []
            target_merged = []
            benchmark_merged = []
            iter_loop = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc="Train", leave=False)
            for i, data in iter_loop:
                epoch_loop.set_description(f'Train: Batch Size={batch_size} | Model={self._type} | lr={self._optimizer.param_groups[0]["lr"]:.8f} | Epoch Loss={epoch_loss:.8f}')
                input, target, benchmark = data
                # print('input type:{}'.format(type(input)))
                # print('input shape:{}'.format(input.shape))
                # print('target type:{}'.format(type(target)))
                # print('target shape:{}'.format(target.shape))
                # print('input:{}'.format(input))
                # print('target:{}'.format(target))
                output = self._net(input)
                # print('output type:{}'.format(type(output)))
                # print('output shape:{}'.format(output.shape))
                # print('benchmark shape:{}'.format(benchmark.shape))
                # print('benchmark type:{}'.format(type(benchmark)))
                self._optimizer.zero_grad()
                loss = self._criterion(output, target)
                loss.backward()
                self._optimizer.step()
                epoch_loss_sum += loss.item()
                for j in range(output.shape[0]):
                    output_merged.append(output[j][-1].item())
                    target_merged.append(target[j][-1].item())
                    benchmark_merged.append(benchmark[j][-1])

            epoch_loss = epoch_loss_sum / total
                        
            if((epoch + 1) % max(1,int(self._total_epoch/self._plot_config['epoch_num_interval'])) == 0):
                result_note = 'train at {} epoch'.format(epoch + 1)
                self._train_comparison_epoch.append((result_note, (output_merged, target_merged, benchmark_merged)))
    
            self._train_loss.append(epoch_loss)
            self.save_net(os.path.join(self._middle_model_dir, 'net_{:.7f}.pth'.format(epoch_loss)))
            self._scheduler.step()
    
    def test(self):
        test_index = 0
        total_loss = 0.0
        file_loop = tqdm(enumerate(self._data_dict.items()), total = len(self._data_dict))
        batch_size = self._data_loader_config['test_batch_num']
        file_loss = 0.0
        for index, (name, test_data) in file_loop:
            total = int(len(test_data) / batch_size)
            file_loss_sum = 0.0
            with torch.no_grad():
                output_merged = []
                target_merged = []
                benchmark_merged = []
                test_dataset = TimeSeriesDataset(test_data)
                test_data_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
                iter_loop = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc="Test", leave=False)
                for i, data in iter_loop:
                    file_loop.set_description(f'Test: Batch Size={batch_size} | Model={self._type} | File Loss={file_loss:.8f}')
                    input, target, benchmark = data
                    output = self._net(input)
                    loss = self._criterion(output, target)
                    for j in range(output.shape[0]):
                        output_merged.append(output[j][-1].item())
                        target_merged.append(target[j][-1].item())
                        benchmark_merged.append(benchmark[j][-1])
                    file_loss_sum += loss.item()
                file_loss = file_loss_sum / total
                self._test_comparison_epoch.append(('{}_{}'.format(name, file_loss),(output_merged, target_merged, benchmark_merged)))
                self._test_loss_epoch.append([x - y for x, y in zip(output_merged, target_merged)])
                total_loss += file_loss
            test_index += 1

    def save_net(self, save_name):
        torch.save(self._net.state_dict(), '{}'.format(save_name))

    def load_net(self):
        self._net.load_state_dict(torch.load('{}/net.pth'.format(self._model_dir), map_location=lambda storage, loc: storage))  
        print("load net successfully!")
        
    def save_result(self):
        if(args.task == 'train'):
            output_dir = os.path.join(root_path, 'output', self._data_loader_config['data_dir'], self._train_config['net_type'], 'train', self._init_time)   
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            self._visualizer.plot_loss(output_dir, self._train_loss)
            self._visualizer.plot_comparison_result(self._train_comparison_epoch, output_dir, 'train_report')
            self._visualizer.plot_comparison_result(self._test_comparison_epoch, output_dir, 'validation_result')
        elif(args.task == 'validation'):
            output_dir = os.path.join(root_path, 'output', self._data_loader_config['data_dir'], self._train_config['net_type'], 'validation', self._init_time)   
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            self._visualizer.plot_comparison_result(self._test_comparison_epoch, output_dir, 'validation_result')
        elif(args.task == 'test'):
            output_dir = os.path.join(root_path, 'output', self._data_loader_config['data_dir'], self._train_config['net_type'], 'test', self._init_time)   
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            self._visualizer.plot_comparison_result(self._test_comparison_epoch, output_dir, 'test_result')
                        
        src_config = os.path.join(root_path,'config',args.config_filepath)
        tgt_config = os.path.join(output_dir,args.config_filepath)
        shutil.copyfile(src_config, tgt_config)

        src_net = os.path.join(root_path, 'model', self._data_loader_config['data_dir'], self._train_config['net_type'],'net.pth')
        tgt_net = os.path.join(output_dir,'net.pth')
        shutil.copyfile(src_net, tgt_net)  
        
               
def main(args):
    config_filepath = os.path.join(root_path,'config',args.config_filepath)
    with open(config_filepath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    time_series_network = TimeSeriesNetwork(config)
    
    if(args.task == 'train'):
        print('============start train============')
        data_dir = os.path.join(args.data_root_dir, config['data_loader']['data_dir'], 'train')
        if not os.path.exists(data_dir):
            raise FileNotFoundError("data dir dose not exit!")
        time_series_network.load_all_data(data_dir)
        
        time_series_network.initialize_model()
        if(config['train']['continue_to_train'] and os.path.exists(os.path.join(root_path, time_series_network.get_model_dir(),'net.pth'))):
            print('continue to train!')
            time_series_network.load_net()
        time_series_network.train()      
        time_series_network.save_net(os.path.join(time_series_network.get_model_dir(), 'net.pth'))

        print('============start validation============')
        data_dir = os.path.join(args.data_root_dir, config['data_loader']['data_dir'], 'validation')
        if not os.path.exists(data_dir):
            raise FileNotFoundError("data dir dose not exit!")
        time_series_network.load_all_data(data_dir)
    else:
        if(args.task == 'validation'):
            data_dir = os.path.join(args.data_root_dir, config['data_loader']['data_dir'], 'validation')
            if not os.path.exists(data_dir):
                raise FileNotFoundError("data dir dose not exit!")
            time_series_network.load_all_data(data_dir)
        elif(args.task == 'test'):
            data_dir = os.path.join(args.data_root_dir, config['data_loader']['data_dir'], 'test')
            if not os.path.exists(data_dir):
                raise FileNotFoundError("data dir dose not exit!")
            time_series_network.load_all_data(data_dir)
        else:
            print('wrong task!')
            
    if not os.path.exists(os.path.join(time_series_network.get_model_dir(),'net.pth')):
        raise FileNotFoundError("the model dose not exit!")
    else:
        time_series_network.initialize_model()
        time_series_network.load_net()
        time_series_network.test()

    time_series_network.save_result()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('RNN Dynamic Modeling Net')
    root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
    parser.add_argument('--config-filepath', default='train_dynamic_model_config.yaml', type=str)
    parser.add_argument('--data-root-dir', default=os.path.join(root_path, 'data'), type=str)
    parser.add_argument('--task', required=True, choices=['train', 'test', 'validation'])
    args = parser.parse_args()
  
    main(args)