#!/usr/bin/env python
from utils.base_data_loader import BaseDataLoader

class FFNNDataLoader(BaseDataLoader):
    def __init__(self, config, device):
        super().__init__(config, device)
        
    def data_postprocess(self, input, output):
        return input.flatten(), output.flatten()
    
    def get_input_dim(self, data_dict):
        name, data = next(iter(data_dict.items()))
        return data[0][0].shape[0]
                      

        