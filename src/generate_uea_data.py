import os
import shutil
from aeon.datasets import load_classification
import pandas as pd
import random
import numpy as np

root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

def generate_uea_data(dataset_name,uea_data_dir):
    X,Y = load_classification(dataset_name)
    print(type(X))
    print(type(Y))
    if(isinstance(X,list)):
        for i in range(len(X)):
            print(X[i].shape) 
    class_mapping = {label: idx for idx, label in enumerate(set(Y))}
    # print(class_mapping)
    # for i in range(len(X)):
    #     print(Y[i]) 
    #     print(X[i])
    #     print('================')    
    # print('X type:{}'.format(type(X)))
    # print('Y type:{}'.format(type(Y)))

    unique_labels, label_counts = np.unique(Y, return_counts=True)

    class_num = len(unique_labels)
    

    X_part1 = []
    Y_part1 = []
    X_part2 = []
    Y_part2 = []
    for label in unique_labels:
        indices = np.where(Y == label)[0]
        N = max(1,int(len(indices) * 0.2))
        selected_indices = np.random.choice(indices, N, replace=False)
        diff_indices = np.setdiff1d(indices, selected_indices)
        for idx in selected_indices:
            # print(Y[idx])
            # print(X[idx])
            X_part1.append(X[idx])
            Y_part1.append(Y[idx])
        # print('***************************************************')
        for idx in diff_indices:
            # print(Y[idx])
            # print(X[idx])
            X_part2.append(X[idx])
            Y_part2.append(Y[idx])
    
    
    # train_output_dir = os.path.join(uea_data_dir, dataset_name,'train','data')
    # if os.path.exists(train_output_dir):
    #     shutil.rmtree(train_output_dir)
    # os.makedirs(train_output_dir)

    # test_output_dir = os.path.join(uea_data_dir, dataset_name,'validation','data')
    # if os.path.exists(test_output_dir):
    #     shutil.rmtree(test_output_dir)
    # os.makedirs(test_output_dir)

    # for i in range(len(X_part2)):
    #     df = pd.DataFrame(X_part2[i].transpose(1,0))
    #     df['ground_truth'] = class_mapping[Y_part2[i]]
    #     df['class_num'] = class_num
    #     df.to_csv(os.path.join(train_output_dir,'{}_train_{}.csv'.format(dataset_name, i)), index=None)

    # for i in range(len(X_part1)):
    #     df = pd.DataFrame(X_part1[i].transpose(1,0))
    #     df['ground_truth'] = class_mapping[Y_part1[i]]
    #     df['class_num'] = class_num
    #     df.to_csv(os.path.join(test_output_dir,'{}_test_{}.csv'.format(dataset_name, i )), index=None)


def download_data():
    # uea_data_filepath = os.path.join(root_path,'data/uea_data/Multivariate/DataDimensions.csv')
    uea_data_filepath = os.path.join(root_path,'data/uea_data/Univariate/SummaryData.csv')
    uea_data = pd.read_csv(uea_data_filepath)
    dataset_names = uea_data['Problem']
    # uea_data_dir = os.path.join(root_path,'data', 'uea_data','Multivariate')
    uea_data_dir = os.path.join(root_path,'data', 'uea_data','Univariate')
    for dataset_name in dataset_names:
        print(dataset_name)
        generate_uea_data(dataset_name,uea_data_dir)

# download_data()
generate_uea_data('AllGestureWiimoteX',os.path.join(root_path,'data', 'uea_data','Univariate'))

# generate_uea_data('StandWalkJump')
# def transfer_uea_data(dataset_name):
#     data_dir = os.path.join(root_path, 'data', 'Multivariate_arff', dataset_name)
#     train_data, meta = arff.loadarff(os.path.join(data_dir,'{}Dimension1_TRAIN.arff'.format(dataset_name)))
#     df = pd.DataFrame(train_data)
#     df.to_csv('a.csv')
#     # print(train_data)
#     print(type(train_data))
#     print(type(meta))
#     print(train_data.shape)
#     # print(meta.shape)

# transfer_uea_data('StandWalkJump')