#!/usr/bin/env python
import json
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
from utils.base_net.fcn import FCN
from utils.base_net.fcn2 import FCN2
from utils.base_net.tcn import TCN
from utils.base_net.ts_transformer import TSTransformer
from utils.base_net.lstm_classification import LSTMClassification
from utils.base_net.base_ffnn import BaseFFNN
from utils.base_net.base_lstm import BaseLSTM
from utils.base_net.base_gru import BaseGRU
from utils.base_net.wavelet_rnn import WaveletRNN
from utils.data_container import DataContainer
from utils.visualizer import Visualizer
from utils.time_series_dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from aeon.datasets import load_classification
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class TrainningFactory:
    def __init__(self, config):
        self._config = config
        self._data_container_config = config['data_container']
        self._train_config = config['train']
        self._device = torch.device('cuda:{}'.format(self._train_config['cuda_id']) if torch.cuda.is_available() else "cpu")
        self._data_dir_name = self._data_container_config['data_dir'].split('/')[-1]
        self._init_time = str(datetime.datetime.now()).replace(' ','_')
        self._output_plot_dir = os.path.join(root_path, 'output', self._data_dir_name, self._init_time)
        self._output_model_dir = os.path.join(root_path, 'model', self._data_dir_name, self._init_time)
        self._lateset_model_dir = os.path.join(root_path, 'model', self._data_dir_name, 'latest_net')
        self._initialized_flag = False

    def load_data(self, data_dir, shuffled):
        data_container = DataContainer(self._data_container_config, self._device)
        data_container.load_data_dict(data_dir)
        data_container.generate_data_loader(shuffled)
        return data_container
        
    def load_all_data(self, data_dir, task):
        self._task = task
        self._data_containers = {}
        if(task == 'train'):
            self._data_containers['train'] = self.load_data(os.path.join(data_dir, 'train'), True)
            self._data_containers['test'] = self.load_data(os.path.join(data_dir, 'validation'), False)
        elif(task == 'validation'):
            self._data_containers['test'] = self.load_data(os.path.join(data_dir, 'validation'), False)
        elif(task == 'test'):
            self._data_containers['test'] = self.load_data(os.path.join(data_dir, 'test'), False)


        print('train data is loaded successfully!')
        
    def initialize_models(self):
        if(not self._initialized_flag):
            self._net_list = []
            for net_type in self._config['train']['net_type_list']:
                net_args_list = self._config['args_dict'][net_type]
                idx = 0
                for net_args in net_args_list:
                    net = TimeSeriesNetwork(self._config, 
                                            net_type, 
                                            idx, 
                                            net_args, 
                                            self._data_containers['test'].get_classification_dict(),
                                            self._data_containers['test'].get_data_shape(),
                                            self._output_model_dir,
                                            self._output_plot_dir)
                    self._net_list.append(net)
                    idx += 1
            self._initialized_flag = True
        else:
            print('Models had been initialized!')

                
    def train(self):
        for net in self._net_list:
            net.train(self._data_containers['train'].get_data_dict(),self._data_containers['test'].get_data_dict())
            net.save_net(os.path.join(net.get_model_dir(), 'net.pth'))
        self.save_lateset_net()
        
    def save_lateset_net(self):
        if os.path.exists(self._lateset_model_dir):
            shutil.rmtree(self._lateset_model_dir)
        shutil.copytree(self._output_model_dir, self._lateset_model_dir)
    
    def get_lateset_model_dir(self):
        return self._lateset_model_dir
    
    def load_nets(self):
        for net in self._net_list:
            net.load_net(self._lateset_model_dir)
            
    def test(self):
        for net in self._net_list:
            net.test(self._data_containers['test'].get_data_dict())
            
    def save_result(self):
        loss_df = pd.DataFrame()
        for net in self._net_list:
            net.save_result()
            if(len(net._test_result) == 0):
                new_df = pd.DataFrame([{'net_type':net._type,
                                        'net_idx':net._idx,
                                        'train_loss':net._final_loss,
                                        'net_args':net._net_args,}])
            else:
                for train_epoch, test_result in net._test_results.items():
                    error_pct = [num / test_result[0] for num in test_result[1]]
                    new_df = pd.DataFrame([{'net_type':net._type,
                                            'net_idx':net._idx,
                                            'net_args':net._net_args,
                                            'train_loss':net._final_loss,
                                            'train_epoch': train_epoch,
                                            'right_pct': error_pct[int(len(error_pct)/2)],
                                            'test_err_pct': error_pct,
                                            'net_args':net._net_args,
                                            }])
                    loss_df = pd.concat([loss_df, new_df], ignore_index=True, sort=False)
        loss_df.to_csv(os.path.join(self._output_plot_dir, 'final_loss.csv'))


class TimeSeriesNetwork:
    def __init__(self, config, net_type, idx, net_args, classification_dict, data_shape, output_model_dir, plot_dir):
        self._data_container_config = config['data_container']
        self._plot_config = config['plot']
        self._train_config = config['train']
        self._type = net_type
        self._idx = idx
        self._net_args = net_args
        self._net_args_json = {
            'model': self._type,
            'index': self._idx,
            'args': self._net_args
        }
        self._classification_dict = classification_dict
        self._feature_num = data_shape['feature_dim']
        self._input_length = data_shape['input_length']
        self._output_length = data_shape['output_length']
        self._device = torch.device('cuda:{}'.format(self._train_config['cuda_id']) if torch.cuda.is_available() else "cpu")
        self._final_loss = 0
        self._test_result = []
        self._train_comparison_epoch = []
        self._test_comparison = []
        self._test_loss_epoch = []
        self._train_loss = []
        self._test_results = OrderedDict()
        self._test_comparisons = OrderedDict()

        self._plot_dir = plot_dir
        self._model_dir = os.path.join(output_model_dir, self._type, str(idx))
        if(args.task == 'train'):
            self._middle_model_dir = os.path.join(self._model_dir, 'middle_model')
            if os.path.exists(self._middle_model_dir):
                shutil.rmtree(self._middle_model_dir)
            os.makedirs(self._middle_model_dir)
        
        self._visualizer = Visualizer(self._plot_config)
        # self._writer = SummaryWriter(os.path.join(root_path, 'logs'))
        self.initialize_model()
               
    def initialize_model(self):
        if(self._data_container_config['classification']):
            if self._type == 'FCN':
                self._total_epoch = self._net_args['total_epoch']
                self._net = FCN(self._feature_num, self._input_length, self._output_length, self._net_args).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._net_args['learning_rate'])
            elif self._type == 'FCN2':
                self._total_epoch = self._net_args['total_epoch']
                self._net = FCN2(self._feature_num, self._input_length, self._output_length, self._net_args).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._net_args['learning_rate'])
            elif self._type == 'TCN':
                self._total_epoch = self._net_args['total_epoch']
                self._net = TCN(self._feature_num, self._input_length, self._output_length, self._net_args).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._net_args['learning_rate'])
            elif self._type == 'LSTMClassification':
                self._total_epoch = self._net_args['total_epoch']
                self._net = LSTMClassification(self._feature_num, self._input_length, self._output_length, self._net_args).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._net_args['learning_rate'])
            elif self._type == 'TSTransformer':
                self._total_epoch = self._net_args['total_epoch']
                self._net = TSTransformer(self._feature_num, self._input_length, self._output_length, self._net_args).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._net_args['learning_rate'])
            else:
                print('{} can not used to classification task!')
        else:
            if self._type == 'LSTM':
                self._total_epoch = self._net_args['total_epoch']
                self._out_dim = self._net_args['out_dim']
                self._hidden_dim = self._net_args['hidden_dim']
                self._num_layers = self._net_args['num_layers']
                self._net = BaseLSTM(self._feature_num, self._out_dim, self._hidden_dim, self._num_layers).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._net_args['learning_rate'])
            elif self._type == 'GRU':
                self._total_epoch = self._net_args['total_epoch']
                self._out_dim = self._net_args['out_dim']
                self._hidden_dim = self._net_args['hidden_dim']
                self._num_layers = self._net_args['num_layers']
                self._net = BaseGRU(self._feature_num, self._out_dim, self._hidden_dim, self._num_layers).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._net_args['learning_rate'])
            elif self._type == 'FFNN':
                self._total_epoch = self._net_args['total_epoch']
                self._out_dim = self._net_args['output_dim']
                self._net = BaseFFNN(self._feature_num, self._net_args['dim_list'], self._out_dim).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._net_args['learning_rate'])
            elif self._type == 'WRNN':
                self._total_epoch = self._net_args['total_epoch']
                self._wrnn_out_dim = self._net_args['wrnn_out_dim']
                self._out_dim = self._net_args['out_dim']
                self._hidden_dim = self._net_args['hidden_dim']
                self._net = WaveletRNN(self._feature_num, self._wrnn_out_dim, self._hidden_dim, self._out_dim, self._device).to(self._device)
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._net_args['learning_rate'])
            else:
                print('{} can not used to predition task!')
        
        if(self._data_container_config['classification']):
            self._criterion = nn.CrossEntropyLoss().to(self._device)
        else:
            self._criterion = nn.MSELoss().to(self._device)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, 
                                                            step_size = self._train_config['scheduler_step_size'], 
                                                            gamma = self._train_config['scheduler_gamma'])
            
    def get_total_epoch(self):
        return self._total_epoch

    def get_model_dir(self):
        return self._model_dir
    
    def train(self, train_data_dict, test_data_dict):
        random.seed(seed)
        data_assembly = []
        for name, data in train_data_dict.items():
            data_assembly.extend(data)

        epoch_loss = 0.0
        epoch_loop = tqdm(range(1, self._total_epoch + 1), 
                          bar_format='{l_bar}|{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining} |{rate_fmt}{postfix}]',
                          leave=True)
        
        for epoch in epoch_loop:
            epoch_loss_sum = 0.0
            output_merged = []
            target_merged = []
            benchmark_merged = []
            random.shuffle(data_assembly)
            batch_size = self._data_container_config['batch_size']
            dataset = TimeSeriesDataset(data_assembly)
            train_data_loader = DataLoader(dataset, 
                                        batch_size = batch_size , 
                                        shuffle = False,
                                        )
            iter_loop = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc="Train", leave=False)
            for i, data in iter_loop:
                epoch_loop.set_description(f'Train: Batch Size={self._data_container_config["batch_size"]} | Model={self._type}_{self._idx} | lr={self._optimizer.param_groups[0]["lr"]:.8f} | Epoch Loss={epoch_loss:.8f}')
                inputs, targets, benchmark = data
                # print('inputs shape:{}'.format(inputs.shape))
                # print('targets shape:{}'.format(targets.shape))
                # print('inputs:{}'.format(inputs))
                # print('targets:{}'.format(targets))
                outputs = self._net(inputs)
                # print('outputs:{}'.format(outputs))
                # print('outputs shape:{}'.format(outputs.shape))
                self._optimizer.zero_grad()
                loss = self._criterion(outputs, targets)
                loss.backward()
                self._optimizer.step()
                epoch_loss_sum += loss.item()
                for j in range(outputs.shape[0]):
                    if(self._data_container_config['classification']):
                        output = outputs[j]
                        target = targets[j]
                        output_id = torch.argmax(output,dim=0)
                        target_id = torch.argmax(target,dim=0)
                        if(not self._data_container_config['use_UEA_data']):
                            output_value = self._classification_dict[output_id.item()]
                            target_value = self._classification_dict[target_id.item()]
                        else:
                            output_value = output_id.item()
                            target_value = target_id.item()
                        # print('output:{}, output_id:{}, output_value:{}'.format(output,output_id,output_value))
                        # print('target:{}, target_id:{}, target_value:{}'.format(target,target_id,target_value))
                        output_merged.append(output_value)
                        target_merged.append(target_value)
                        benchmark_merged.append(target_value)
                    else:
                        output_merged.append(output[j][-1].item())
                        target_merged.append(target[j][-1].item())
                        benchmark_merged.append(benchmark[j][-1])

            epoch_loss = epoch_loss_sum / len(train_data_loader)
            if((epoch == 1) or (epoch % self._plot_config['epoch_interval_for_result'] == 0) or (epoch == self._total_epoch)):
                result_note = 'train at {} epoch'.format(epoch)
                self._train_comparison_epoch.append((result_note, (output_merged, target_merged, benchmark_merged)))

            if(self._train_config['enable_test_when_train'] and ((epoch == 1) or (epoch % self._train_config['epoch_interval_for_test'] == 0) or (epoch == self._total_epoch))):
                self.test(test_data_dict, epoch)
                    
            self._train_loss.append(epoch_loss)
            self.save_net(os.path.join(self._middle_model_dir, 'net_{:.7f}.pth'.format(epoch_loss)))
            self._scheduler.step()
        self._final_loss = epoch_loss
    
    def test(self, data_dict, train_epoch = 0):
        file_loop = tqdm(enumerate(data_dict.items()), total = len(data_dict))
        batch_size = self._data_container_config['batch_size']
        file_loss = 0.0
        self._test_comparison = []
        if(self._data_container_config['classification']):
            error_range = 5
            idx_offset = int(error_range / 2)
            self._test_result = [0] * error_range
            self._test_num = 0
        for index, (name, test_data) in file_loop:
            total = max(int(len(test_data) / batch_size),1)
            file_loss_sum = 0.0
            with torch.no_grad():
                output_merged = []
                target_merged = []
                benchmark_merged = []
                test_dataset = TimeSeriesDataset(test_data)
                test_data_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
                iter_loop = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc="Test", leave=False)
                for i, data in iter_loop:
                    file_loop.set_description(f'Test: Batch Size={batch_size} | Model={self._type}_{self._idx} | train_epoch={train_epoch} | File Loss={file_loss:.8f}')
                    inputs, targets, benchmark = data
                    outputs = self._net(inputs)
                    loss = self._criterion(outputs, targets)
                    for j in range(outputs.shape[0]):
                        if(self._data_container_config['classification']):
                            output = outputs[j]
                            target = targets[j]
                            output_id = torch.argmax(output,dim=0)
                            target_id = torch.argmax(target,dim=0)
                            if(not self._data_container_config['use_UEA_data']):
                                output_value = self._classification_dict[output_id.item()]
                                target_value = self._classification_dict[target_id.item()]
                            else:
                                output_value = output_id.item()
                                target_value = target_id.item()
                            output_merged.append(output_value)
                            target_merged.append(target_value)
                            benchmark_merged.append(target_value)
                            error = (target_id - output_id).item()
                            self._test_num += 1
                            if(error <= idx_offset and error >= - idx_offset):
                                self._test_result[error + idx_offset] += 1
                        else:
                            output_merged.append(output[j][-1].item())
                            target_merged.append(target[j][-1].item())
                            benchmark_merged.append(benchmark[j][-1])
                    file_loss_sum += loss.item()
                file_loss = file_loss_sum / total
                self._test_comparison.append(('{}_{}'.format(name, file_loss),(output_merged, target_merged, benchmark_merged)))
        self._test_results[train_epoch] = (self._test_num, self._test_result)
        self._test_comparisons[train_epoch] = self._test_comparison

    def save_net(self,save_name):
        torch.save(self._net.state_dict(), '{}'.format(save_name))

    def load_net(self, latest_model_dir):
        latest_model_dir = os.path.join(latest_model_dir, self._type, str(self._idx))
        self._net.load_state_dict(torch.load('{}/net.pth'.format(latest_model_dir), map_location=lambda storage, loc: storage))  
        print("load net successfully!")
        
    def save_result(self):
        output_dir = os.path.join(self._plot_dir, self._type, str(self._idx))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print("output dir : {}".format(output_dir))
        if(args.task == 'train'):
            self._visualizer.plot_loss(output_dir, self._train_loss)
            self._visualizer.plot_comparison_result(self._train_comparison_epoch, output_dir, 'train_result')

            for train_epoch, test_comparison in self._test_comparisons.items():
                test_comparison_output_dir = os.path.join(output_dir, 'validation_result','train_epoch={}'.format(train_epoch))
                os.makedirs(test_comparison_output_dir)
                self._visualizer.plot_comparison_result(test_comparison, test_comparison_output_dir, 'validation_result')

            src_config = os.path.join(root_path,'config',args.config_filepath)
            tgt_config = os.path.join(output_dir,args.config_filepath)
            shutil.copyfile(src_config, tgt_config)
            
            with open(os.path.join(output_dir,'net_args.json'), 'w') as f:
                json.dump(self._net_args_json, f)
                
            src_net = os.path.join(self._model_dir,'net.pth')
            tgt_net = os.path.join(output_dir,'net.pth')
            shutil.copyfile(src_net, tgt_net)
            
            src_middle_net = os.path.join(self._model_dir,'middle_model')
            tgt_middle_net = os.path.join(output_dir,'middle_model')
            shutil.copytree(src_middle_net, tgt_middle_net)
        
        elif(args.task == 'validation'):
            for train_epoch, test_comparison in self._test_comparisons.items():
                test_comparison_output_dir = os.path.join(output_dir, 'train_epoch={}'.format(train_epoch))
                os.makedirs(test_comparison_output_dir)
                self._visualizer.plot_comparison_result(test_comparison, test_comparison_output_dir, 'validation_result')
        elif(args.task == 'test'):
            for train_epoch, test_comparison in self._test_comparisons.items():
                test_comparison_output_dir = os.path.join(output_dir, 'train_epoch={}'.format(train_epoch))
                os.makedirs(test_comparison_output_dir)
                self._visualizer.plot_comparison_result(test_comparison, test_comparison_output_dir, 'test_result')
        
               
def main(args):
    config_filepath = os.path.join(root_path,'config',args.config_filepath)
    with open(config_filepath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    traning_factory = TrainningFactory(config)

    data_dir = os.path.join(config['data_container']['data_dir'])
    if not os.path.exists(data_dir):
        raise FileNotFoundError("data dir dose not exit!")
     
    if(args.task == 'train'):
        traning_factory.load_all_data(data_dir, args.task)
        traning_factory.initialize_models()
        traning_factory.train()
        # traning_factory.test()
    else:
        traning_factory.load_all_data(data_dir, args.task)
        traning_factory.initialize_models()
        traning_factory.load_nets()
        traning_factory.test()

    traning_factory.save_result()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('RNN Dynamic Modeling Net')
    root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
    parser.add_argument('--config-filepath', default='train_weight_estimation_config.yaml', type=str)
    # parser.add_argument('--config-filepath', default='train_dynamic_model_config.yaml', type=str)
    parser.add_argument('--task', default = 'train', choices=['train', 'test', 'validation'])
    
    args = parser.parse_args()
  
    main(args)