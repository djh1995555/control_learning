#!/usr/bin/env python
import os
import torch
import numpy as np
import pandas as pd


class BaseDataLoader():
    def __init__(self, config, device):
        self._config = config
        self._device = device
        self._is_normalized = False
        
    def load_data_dict(self, data_dir, data_range, validation = False):
        data_dir = os.path.join(data_dir, 'data')
        print('data_dir = {}'.format(data_dir))
        data_dict = {}
        self._data_dir = data_dir
        file_list = list(os.listdir(data_dir))
        file_list = sorted(file_list)
        for file_name in file_list:
            # print('file_name : {}'.format(file_name))
            name, extension = os.path.splitext(file_name)
            file_path = os.path.join(data_dir, file_name)
            if (extension == '.csv'):
                raw_data = pd.read_csv(file_path)
                data_list = self.load_data(name, raw_data, data_range, validation)
                data_dict[name] = data_list
                # print('load data of {}'.format(name))
        return data_dict, self.get_input_dim(data_dict)
    
    def get_std(self):
        return self._std
    
    def get_mean(self):
        return self._mean
    
    def load_data(self, name, data, data_range, validation = False):
        data_y = np.array(data[self._config['ground_truth']])
        benchmark = np.array(data[self._config['benchmark']])
        drop_columns = [x for x in self._config['drop_columns']]
        data_x = np.array(data.drop(drop_columns, axis = 1))
        if(self._config['normalized']): 
            if(not self._is_normalized): 
                self._mean = data_x.mean(axis=0)
                self._std = data_x.std(axis=0)
                self._is_normalized = True
            data_x = np.where(self._std == 0, 0, (data_x - self._mean) / self._std)

        offset = int(self._config['gt_offset'] / self._config['sample_time'])
        if(len(data_range) == 0):
            data_x = data_x[0 : - offset]
            data_y = data_y[offset : -1]
            benchmark = np.array(data[self._config['ground_truth']])[0 : - offset]
        else:
            data_range[0] = max(data_range[0],0)
            data_range[1] = min(data_range[1], data_x.shape[0]-offset)
            data_x = data_x[data_range[0] : data_range[1]]
            data_y = data_y[data_range[0] + offset : data_range[1] + offset] 
            benchmark = np.array(data[self._config['ground_truth']])[data_range[0] : data_range[1]]
        
        self._sequence_length = int(self._config['sequence_duration'] * self._config['frequency'])
        self._step_length = int(self._config['step'] * self._config['frequency'])
        
        return self.orgnize_data(data_x, data_y, benchmark)
    def orgnize_data(self, data_x, data_y, benchmark):
        data_list = []
        index = 0
        while index + self._sequence_length <= data_x.shape[0]: 
            single_input = data_x[index : index + self._sequence_length, : ]
            single_output = data_y[index : index + self._sequence_length]
            single_benchmark = benchmark[index : index + self._sequence_length]

            single_input, single_output = self.data_postprocess(single_input, single_output)
            single_input = torch.tensor(single_input, dtype = torch.float32, device = self._device)
            single_output = torch.tensor(single_output, dtype = torch.float32, device = self._device) 
            # print('before intput shape:{}, output shape:{}'.format(single_input.shape, single_output.shape))
            data_list.append((single_input,single_output, single_benchmark))
            index += self._step_length
        return data_list
    
    def data_postprocess(self, input, output):
        pass

    def get_input_dim(self, data_dict):
        pass
                      

        