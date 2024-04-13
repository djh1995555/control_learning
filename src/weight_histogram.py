import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder_path = "/home/jianhao.dong/l4e2/data_driven/control_learning/weight_gt_record"


all_data = []


for filename in os.listdir(folder_path):
    if filename.endswith(".csv") and 'pdb' in filename:
        print(filename)
        file_path = os.path.join(folder_path, filename)
        
        df = pd.read_csv(file_path)
        df = df.dropna()
        weights = np.floor(df['weight_gt'] / 2) * 2
        hist, bins = np.histogram(weights, bins=20)
        plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]), align='edge')
        plt.title('Histogram of weight_gt')
        plt.xlabel('weight_gt')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(folder_path, 'result_{}.png'.format(filename)))
        plt.close()


