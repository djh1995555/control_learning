#!/usr/bin/env python
from utils.base_data_loader import BaseDataLoader
import numpy as np

class RNNDataLoader(BaseDataLoader):
    def __init__(self, config, device):
        super().__init__(config, device)

    def data_postprocess(self, input, output):
        return input[ :, :], output[ :, np.newaxis]
    
    def get_input_dim(self, data_dict):
        name, data = next(iter(data_dict.items()))
        print('shape:{}'.format(data[0][0].shape))
        return data[0][0].shape[1]
                      

        