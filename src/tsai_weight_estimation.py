from tsai.all import *
import sklearn.metrics as skm
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

my_setup()
model_mapping = {
    'FCN': FCN,
    'FCNPlus': FCNPlus,
    'InceptionTime': InceptionTime,
    'InceptionTimePlus': InceptionTimePlus,
    'InCoordTime': InCoordTime,
    'XCoordTime': XCoordTime,
    'InceptionTimePlus17x17': InceptionTimePlus17x17,
    'InceptionTimePlus32x32': InceptionTimePlus32x32,
    'InceptionTimePlus47x47': InceptionTimePlus47x47,
    'InceptionTimePlus62x62': InceptionTimePlus62x62,
    'InceptionTimeXLPlus': InceptionTimeXLPlus,
    'MultiInceptionTimePlus': MultiInceptionTimePlus,
    'MiniRocket': MiniRocket,
    'MiniRocketRegressor': MiniRocketRegressor,
    'MiniRocketVotingClassifier': MiniRocketVotingClassifier,
    'MiniRocketVotingRegressor': MiniRocketVotingRegressor,
    'MiniRocket': MiniRocket,
    'MiniRocketPlus': MiniRocketPlus,
    'MultiRocket': MultiRocket,
    'MultiRocketPlus': MultiRocketPlus,
    'InceptionRocketPlus': InceptionRocketPlus,
    'MLP': MLP,
    'gMLP': gMLP,
    'MultiInputNet': MultiInputNet,
    'OmniScaleCNN': OmniScaleCNN,
    'RNN': RNN,
    'LSTM': LSTM,
    'GRU': GRU,
    'RNNPlus': RNNPlus,
    'LSTMPlus': LSTMPlus,
    'GRUPlus': GRUPlus,
    'RNN_FCN': RNN_FCN,
    'LSTM_FCN': LSTM_FCN,
    'GRU_FCN': GRU_FCN,
    'MRNN_FCN': MRNN_FCN,
    'MLSTM_FCN': MLSTM_FCN,
    'MGRU_FCN': MGRU_FCN,
    'RNN_FCNPlus': RNN_FCNPlus,
    'LSTM_FCNPlus': LSTM_FCNPlus,
    'GRU_FCNPlus': GRU_FCNPlus,
    'MRNN_FCNPlus': MRNN_FCNPlus,
    'MLSTM_FCNPlus': MLSTM_FCNPlus,
    'MGRU_FCNPlus': MGRU_FCNPlus,
    'ROCKET': ROCKET,
    'RocketClassifier': RocketClassifier,
    'RocketRegressor': RocketRegressor,
    'ResCNN': ResCNN,
    'ResNet': ResNet,
    'ResNetPlus': ResNetPlus,
    'TCN': TCN,
    'TSPerceiver': TSPerceiver,
    'TST': TST,
    'TSTPlus': TSTPlus,
    'MultiTSTPlus': MultiTSTPlus,
    'TSiT': TSiT,
    'TSiTPlus': TSiTPlus,
    'TabFusionTransformer': TabFusionTransformer,
    'TSTabFusionTransformer': TSTabFusionTransformer,
    'TabModel': TabModel,
    'TabTransformer': TabTransformer,
    'GatedTabTransformer': GatedTabTransformer,
    'TransformerModel': TransformerModel,
    'XCM': XCM,
    'XCMPlus': XCMPlus,
    'xresnet1d18': xresnet1d18,
    'xresnet1d34': xresnet1d34,
    'xresnet1d50': xresnet1d50,
    'xresnet1d101': xresnet1d101,
    'xresnet1d152': xresnet1d152,
    'xresnet1d18_deep': xresnet1d18_deep,
    'xresnet1d34_deep': xresnet1d34_deep,
    'xresnet1d50_deep': xresnet1d50_deep,
    'xresnet1d18_deeper': xresnet1d18_deeper,
    'xresnet1d34_deeper': xresnet1d34_deeper,
    'xresnet1d50_deeper': xresnet1d50_deeper,
    'XResNet1dPlus': XResNet1dPlus,
    'xresnet1d18plus': xresnet1d18plus,
    'xresnet1d34plus': xresnet1d34plus,
    'xresnet1d50plus': xresnet1d50plus,
    'xresnet1d101plus': xresnet1d101plus,
    'xresnet1d152plus': xresnet1d152plus,
    'xresnet1d18_deepplus': xresnet1d18_deepplus,
    'xresnet1d34_deepplus': xresnet1d34_deepplus,
    'xresnet1d50_deepplus': xresnet1d50_deepplus,
    'xresnet1d18_deeperplus': xresnet1d18_deeperplus,
    'xresnet1d34_deeperplus': xresnet1d34_deeperplus,
    'xresnet1d50_deeperplus': xresnet1d50_deeperplus,
    'XceptionTime': XceptionTime,
    'XceptionTimePlus': XceptionTimePlus,
    'mWDN': mWDN,
    'mWDNPlus': mWDNPlus,
    'TSSequencer': TSSequencer,
    'TSSequencerPlus': TSSequencerPlus,
    "PatchTST": PatchTST,
    "ConvTran": ConvTran,
    "ConvTranPlus": ConvTranPlus,
    "RNNAttention": RNNAttention,
    "LSTMAttention": LSTMAttention, 
    "GRUAttention": GRUAttention,
    # "RNNAttentionPlus": RNNAttentionPlus, 
    # "LSTMAttentionPlus": LSTMAttentionPlus,
    # "GRUAttentionPlus": GRUAttentionPlus,
    "TransformerRNNPlus": TransformerRNNPlus, 
    "TransformerLSTMPlus": TransformerLSTMPlus,
    "TransformerGRUPlus": TransformerGRUPlus, 
    "Hydra": Hydra, 
    "HydraPlus": HydraPlus,
    "HydraMultiRocket": HydraMultiRocket, 
    "HydraMultiRocketPlus": HydraMultiRocketPlus,
}

def get_data_list(data_array, data_y, seq_len):
    data_x_list = []
    data_y_list = []
    index = 0
    while index + seq_len <= data_array.shape[1]: 
        data_x = data_array[:, index : index + seq_len]
        index += seq_len
        data_x_list.append(data_x)
        data_y_list.append(data_y)
    return data_x_list, data_y_list

def load_data(data_dir, split_ratio = 0.75):
    file_list = list(os.listdir(data_dir))
    file_list = sorted(file_list)
    data = []
    label = []
    is_normalized = False
    mean = 0.0
    std = 0.0
    interval_length = 4
    seq_len = 40
    for file_name in file_list:
        name, extension = os.path.splitext(file_name)
        file_path = os.path.join(data_dir, file_name)
        if (extension == '.csv'):
            raw_data = pd.read_csv(file_path)
            weight_gt = str(int(raw_data['weight'].mean()/interval_length)*interval_length)

            raw_data = raw_data.drop(['weight','benchmark','steering_angle'], axis=1)
            data_array = raw_data.values
            
            if(not is_normalized): 
                mean = data_array.mean(axis=0)
                std = data_array.std(axis=0)
                is_normalized = True
            
            data_array = np.where(std == 0, 0, (data_array - mean) / std)
            data_df = pd.DataFrame(data_array)
            df_file_path = os.path.join('./data1', file_name)
            data_df.to_csv(df_file_path)
            data_array = data_array.transpose()
            data_x_list, data_y_list = get_data_list(data_array, weight_gt, seq_len)
            data.extend(data_x_list)
            label.extend(data_y_list)
    
    x = np.array(data)
    print('x shape:{}'.format(x.shape))
    y = np.array(label)
    print('y shape:{}'.format(y.shape))

    unique_labels, label_counts = np.unique(y, return_counts=True)
    train_idx = []
    test_idx = []
    for label in unique_labels:
        idx_for_one_class = np.where(y == label)[0]
        N = max(1,int(len(idx_for_one_class) * split_ratio))
        train_idx_for_one_class = np.random.choice(idx_for_one_class, N, replace=False)
        test_idx_for_one_class = np.setdiff1d(idx_for_one_class, train_idx_for_one_class)
        train_idx.extend(train_idx_for_one_class)
        test_idx.extend(test_idx_for_one_class)

    return x, y, (train_idx, test_idx)

data_dir = '/home/jianhao_dong/NN_weight_estimator/data/weight_estimation_tsai'
# data_dir = '/home/jianhao_dong/NN_weight_estimator/data/weight_tsai'
x, y, splits = load_data(data_dir)

csv_file_path = 'result.csv'
tfms = [None, Categorize()]
batch_tfms = TSStandardize()
dls = get_ts_dls(x, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=8)
dls.show_batch(sharey=True)
dsid = 'weight_estimation'
all_arch_names =  [
                # 'FCN', 'FCNPlus', 
                # 'InceptionTime', 'InceptionTimePlus', 'InCoordTime', 'XCoordTime', 
                # 'InceptionTimePlus17x17', 'InceptionTimePlus32x32', 'InceptionTimePlus47x47', 
                # 'InceptionTimePlus62x62',  
                # 'MultiInceptionTimePlus', 
                # 'MultiRocket', 'MultiRocketPlus', 'InceptionRocketPlus', 
                # 'MLP', 
                # 'OmniScaleCNN', 
                # 'RNN', 'LSTM', 'GRU', 
                # 'RNNPlus', 'LSTMPlus', 'GRUPlus', 'RNN_FCN', 'LSTM_FCN', 'GRU_FCN', 'MRNN_FCN', 'MLSTM_FCN', 'MGRU_FCN', 
                # 'RNN_FCNPlus', 'LSTM_FCNPlus', 'GRU_FCNPlus', 'MRNN_FCNPlus', 'MLSTM_FCNPlus', 'MGRU_FCNPlus', 
                # 'ResCNN', 'ResNet', 'ResNetPlus', 'TCN', 
                # 'XCM', 'XCMPlus', 
                # 'xresnet1d18', 'xresnet1d34', 'xresnet1d50', 'xresnet1d101', 'xresnet1d152', 'xresnet1d18_deep', 'xresnet1d34_deep', 'xresnet1d50_deep', 
                # 'xresnet1d18_deeper', 'xresnet1d34_deeper', 'xresnet1d50_deeper', 
                
                # 'xresnet1d18plus', 'xresnet1d34plus', 
                # 'xresnet1d50plus', 'xresnet1d101plus', 
                # 'xresnet1d152plus', 'xresnet1d18_deepplus', 'xresnet1d34_deepplus', 'xresnet1d50_deepplus', 
                # 'xresnet1d18_deeperplus', 'xresnet1d34_deeperplus', 'xresnet1d50_deeperplus', 'XceptionTime', 'XceptionTimePlus', 'mWDN', 'mWDNPlus',
                # 'TSSequencer', 
                # 'TSSequencerPlus',
                # "ConvTran", "ConvTranPlus",
                # "RNNAttention", "LSTMAttention", "GRUAttention", 
                # "TransformerRNNPlus", "TransformerLSTMPlus", "TransformerGRUPlus", "Hydra", "HydraPlus", "HydraMultiRocket", "HydraMultiRocketPlus",

                # 'TabFusionTransformer', 'TSTabFusionTransformer', 'TabModel', 'TabTransformer', 'GatedTabTransformer', 'TransformerModel', 
                # "MiniRocket", 'MiniRocketClassifier','MiniRocketRegressor', 'MiniRocketVotingClassifier', 'MiniRocketVotingRegressor', 'MiniRocketPlus',
                # 'ROCKET', 'RocketClassifier', 'RocketRegressor',
                # 'MultiInputNet',

                # 'gMLP', 'InceptionTimeXLPlus','TSPerceiver', 
                # 'TST', 'TSTPlus', 'MultiTSTPlus', 
                # 'TSiT', 'TSiTPlus', 
                # "PatchTST", 
                # "RNNAttentionPlus", "LSTMAttentionPlus", "GRUAttentionPlus", 
                # 'XResNet1dPlus', 
                ]
# all_arch_names =  ['FCN','FCNPlus']

re = pd.DataFrame()
for model_name in all_arch_names:
    print(f'current model is {model_name}')
    model = model_mapping[model_name]
    learn = ts_learner(dls, model, c_in=dls.vars, c_out=dls.c, seq_len=dls.len, metrics=accuracy)
    learn.lr_find()
    learn.fit_one_cycle(5, lr_max=1e-3)
    final_record = learn.final_record
    df = pd.DataFrame([{
                    'dataset':dsid,
                    'model_name':model_name, 
                    'train_loss':final_record[0],
                    'valid_loss':final_record[1],
                    'accuracy':final_record[2],
                    }])
    re = pd.concat([re,df], ignore_index=True, sort=False)
    if not os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, index=False)
    else:
        df.to_csv(csv_file_path, mode='a', header=False, index=False)    
    
    del model
    del learn
    torch.cuda.empty_cache() 
re.to_csv('final_result.csv')