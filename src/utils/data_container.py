#!/usr/bin/env python
import os
import glob
import torch
import numpy as np
import pandas as pd
from utils.time_series_dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import random
import pickle


seed = 999
np.random.seed(seed)
random.seed(seed)

class DataContainer():
    def __init__(self, config, device):
        self._config = config
        self._device = device
        self._is_normalized = False

    def one_hot_encode(self, sequence):
        num_classes = len(set(sequence))
        
        class_mapping = {label: idx for idx, label in enumerate(set(sequence))}
        self._classification_dict = {idx:label for idx, label in enumerate(set(sequence))}
        
        one_hot_vectors = []
        for label in sequence:
            vector = np.zeros(num_classes)
            vector[class_mapping[label]] = 1
            one_hot_vectors.append(vector)
        
        return np.array(one_hot_vectors)

    def load_data_dict(self, data_dir):
        data_dir = os.path.join(data_dir, 'data')
        print('data_dir = {}'.format(data_dir))
        self._classification_dict = self.generate_classification_dict(self._config['classification_range'], self._config['classification_interval'])
        self._data_dir = data_dir
        pkl_files = glob.glob(os.path.join(self._data_dir, '*.pkl'))

        if pkl_files:
            for pkl_file in pkl_files:
                print('read from pikcle!')
                with open(pkl_file, 'rb') as file:
                    self._data_dict = {}
                    self._data_dict = pickle.load(file)
        else:
            self._data_dict = {}
            file_list = list(os.listdir(self._data_dir))
            file_list = sorted(file_list)
            for file_name in file_list:
                name, extension = os.path.splitext(file_name)
                file_path = os.path.join(self._data_dir, file_name)
                if (extension == '.csv'):
                    raw_data = pd.read_csv(file_path)
                    data_list = self.load_data(raw_data, self._config['data_range'])
                    self._data_dict[name] = data_list
            # with open(os.path.join(self._data_dir, 'data_dict.pkl'), 'wb') as file:
            #     print('save as pikcle!')
            #     pickle.dump(self._data_dict, file) 
                                   
        all_values = self._data_dict.values()
        all_values_list = list(all_values)
        self._input_length = all_values_list[0][0][0].shape[0]
        self._feature_dim = all_values_list[0][0][0].shape[1]
        self._output_length = all_values_list[0][0][1].shape[0]
        self._data_shape = {'feature_dim':self._feature_dim, 'input_length':self._input_length, 'output_length':self._output_length}

    def load_data(self, data, data_range):
        data_y = np.array(data[self._config['ground_truth']])
        benchmark = np.array(data[self._config['benchmark']])
        data_x = data.loc[:, self._config['involved_columns']]
        if(self._config['normalized']): 
            if(not self._is_normalized): 
                self._mean = data_x.mean(axis=0)
                self._std = data_x.std(axis=0)
                self._is_normalized = True
            data_x = np.where(self._std == 0, 0, (data_x - self._mean) / self._std)
        if(self._config['gt_offset'] > 0.0):
            offset = int(self._config['gt_offset'] / self._config['sample_time'])
            if(len(data_range) == 0):
                data_x = data_x[0 : - offset]
                data_y = data_y[offset : -1]
                benchmark = np.array(data[self._config['ground_truth']])[0 : - offset]
            else:
                data_range_start = max(data_range[0],0)
                data_range_end = min(data_range[1], data_x.shape[0]-offset)
                data_x = data_x[data_range_start : data_range_end]
                data_y = data_y[data_range_start + offset : data_range_end + offset] 
                benchmark = np.array(data[self._config['ground_truth']])[data_range_start : data_range_end]

        frequency = 1 / self._config['sample_time']
        self._sequence_length = int(self._config['sequence_duration'] * frequency)
        self._step_length = int(self._config['step'] * frequency)
        return self.orgnize_data(data_x, data_y, benchmark)
    
    def orgnize_data(self, data_x, data_y, benchmark):
        data_list = []
        index = 0
        while index + self._sequence_length <= data_x.shape[0]: 
            single_input = data_x[index : index + self._sequence_length, : ]
            single_output = data_y[index : index + self._sequence_length]
            if(self._config['classification']):
                avg_value = single_output.sum()/single_output.shape[0]
                single_output = self.map_to_intervals(avg_value, self._config['classification_range'], self._config['classification_interval'])
            single_benchmark = benchmark[index : index + self._sequence_length]

            single_input, single_output = self.data_postprocess(single_input, single_output)
            single_input = torch.tensor(single_input, dtype = torch.float32, device = self._device)
            single_output = torch.tensor(single_output, dtype = torch.float32, device = self._device)
            data_list.append((single_input,single_output, single_benchmark))
            # print('single_input shape:{}'.format(single_input.shape))
            # print('single_output shape:{}'.format(single_output.shape))
            index += self._step_length
        return data_list
    

    def generate_data_loader(self, shuffled = False):
        data_assembly = []
        for name, data in self._data_dict.items():
            data_assembly.extend(data)
        if(shuffled):
            random.shuffle(data_assembly)
        batch_size = self._config['batch_size']
        dataset = TimeSeriesDataset(data_assembly)
        self._data_loader = DataLoader(dataset, 
                                       batch_size = batch_size , 
                                       shuffle = False,
                                       )
    
    def get_data_loader(self):
        return self._data_loader
    
    def get_data_shape(self):
        return self._data_shape

    def get_data_dict(self):
        return self._data_dict

    def get_classification_dict(self):
        return self._classification_dict
        
    def get_std(self):
        return self._std
    
    def get_mean(self):
        return self._mean

    def generate_classification_dict(self, class_range, m):
        a = class_range[0]
        b = class_range[1]
        N = int((b - a) / m) + 1
        classification_dict = {}
        for i in range(N):
            classification_dict[i] = a + i * m
        return classification_dict
            
    def map_to_intervals(self, data, class_range, m):
        a = class_range[0]
        b = class_range[1]
        N = int((b - a) / m) + 1
        mapped_data = ((data - a) / m).astype(int)
        result_array = np.eye(N)[mapped_data]
        return result_array
    
    def data_postprocess(self, input, output):
        return input[ :, :], output[ :, np.newaxis]
    
                      

        