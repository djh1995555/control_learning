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
# model_mapping = {
#     'FCN': FCN(nvars, c_out),
#     'FCNPlus': FCNPlus(nvars, c_out),
#     'InceptionTime': InceptionTime(nvars, c_out),
#     'InceptionTimePlus': InceptionTimePlus(nvars, c_out),
#     'InCoordTime': InCoordTime(nvars, c_out),
#     'XCoordTime': XCoordTime(nvars, c_out),
#     'InceptionTimePlus17x17': InceptionTimePlus17x17(nvars, c_out),
#     'InceptionTimePlus32x32': InceptionTimePlus32x32(nvars, c_out),
#     'InceptionTimePlus47x47': InceptionTimePlus47x47(nvars, c_out),
#     'InceptionTimePlus62x62': InceptionTimePlus62x62(nvars, c_out),
#     'InceptionTimeXLPlus': InceptionTimeXLPlus(nvars, c_out),
#     'MultiInceptionTimePlus': MultiInceptionTimePlus(nvars, c_out),
    
#     'MultiRocket': MultiRocket(nvars, c_out, seq_len),
#     'MultiRocketPlus': MultiRocketPlus(nvars, c_out, seq_len),
#     'InceptionRocketPlus': InceptionRocketPlus(nvars, c_out, seq_len),
#     'MLP': MLP(nvars, c_out, seq_len),
#     'gMLP': gMLP(nvars, c_out, seq_len),
#     'OmniScaleCNN': OmniScaleCNN(nvars, c_out, seq_len),
#     'RNN': RNN(nvars, c_out),
#     'LSTM': LSTM(nvars, c_out),
#     'GRU': GRU(nvars, c_out),
#     'RNNPlus': RNNPlus(nvars, c_out, seq_len),
#     'LSTMPlus': LSTMPlus(nvars, c_out, seq_len),
#     'GRUPlus': GRUPlus(nvars, c_out, seq_len),
#     'RNN_FCN': RNN_FCN(nvars, c_out, seq_len),
#     'LSTM_FCN': LSTM_FCN(nvars, c_out, seq_len),
#     'GRU_FCN': GRU_FCN(nvars, c_out, seq_len),
#     'MRNN_FCN': MRNN_FCN(nvars, c_out, seq_len),
#     'MLSTM_FCN': MLSTM_FCN(nvars, c_out, seq_len),
#     'MGRU_FCN': MGRU_FCN(nvars, c_out, seq_len),
#     'RNN_FCNPlus': RNN_FCNPlus(nvars, c_out, seq_len),
#     'LSTM_FCNPlus': LSTM_FCNPlus(nvars, c_out, seq_len),
#     'GRU_FCNPlus': GRU_FCNPlus(nvars, c_out, seq_len),
#     'MRNN_FCNPlus': MRNN_FCNPlus(nvars, c_out, seq_len),
#     'MLSTM_FCNPlus': MLSTM_FCNPlus(nvars, c_out, seq_len),
#     'MGRU_FCNPlus': MGRU_FCNPlus(nvars, c_out, seq_len),
#     'ResCNN': ResCNN(nvars, c_out),
#     'ResNet': ResNet(nvars, c_out),
#     'ResNetPlus': ResNetPlus(nvars, c_out, seq_len),
#     'TCN': TCN(nvars, c_out),
#     'TSPerceiver': TSPerceiver(nvars, c_out, seq_len),
#     'TST': TST(nvars, c_out, seq_len),
#     'TSTPlus': TSTPlus(nvars, c_out, seq_len),
#     'MultiTSTPlus': MultiTSTPlus(nvars, c_out, seq_len),
#     'TSiT': TSiT(nvars, c_out, seq_len),
#     'TSiTPlus': TSiTPlus(nvars, c_out, seq_len),
#     'TransformerModel': TransformerModel(nvars, c_out),
#     'XCM': XCM(nvars, c_out, seq_len),
#     'XCMPlus': XCMPlus(nvars, c_out, seq_len),
#     'xresnet1d18': xresnet1d18(nvars, c_out),
#     'xresnet1d34': xresnet1d34(nvars, c_out),
#     'xresnet1d50': xresnet1d50(nvars, c_out),
#     'xresnet1d101': xresnet1d101(nvars, c_out),
#     'xresnet1d152': xresnet1d152(nvars, c_out),
#     'xresnet1d18_deep': xresnet1d18_deep(nvars, c_out),
#     'xresnet1d34_deep': xresnet1d34_deep(nvars, c_out),
#     'xresnet1d50_deep': xresnet1d50_deep(nvars, c_out),
#     'xresnet1d18_deeper': xresnet1d18_deeper(nvars, c_out),
#     'xresnet1d34_deeper': xresnet1d34_deeper(nvars, c_out),
#     'xresnet1d50_deeper': xresnet1d50_deeper(nvars, c_out),
#     'XResNet1dPlus': XResNet1dPlus(nvars, c_out),
#     'xresnet1d18plus': xresnet1d18plus(nvars, c_out),
#     'xresnet1d34plus': xresnet1d34plus(nvars, c_out),
#     'xresnet1d50plus': xresnet1d50plus(nvars, c_out),
#     'xresnet1d101plus': xresnet1d101plus(nvars, c_out),
#     'xresnet1d152plus': xresnet1d152plus(nvars, c_out),
#     'xresnet1d18_deepplus': xresnet1d18_deepplus(nvars, c_out),
#     'xresnet1d34_deepplus': xresnet1d34_deepplus(nvars, c_out),
#     'xresnet1d50_deepplus': xresnet1d50_deepplus(nvars, c_out),
#     'xresnet1d18_deeperplus': xresnet1d18_deeperplus(nvars, c_out),
#     'xresnet1d34_deeperplus': xresnet1d34_deeperplus(nvars, c_out),
#     'xresnet1d50_deeperplus': xresnet1d50_deeperplus(nvars, c_out),
#     'XceptionTime': XceptionTime(nvars, c_out),
#     'XceptionTimePlus': XceptionTimePlus(nvars, c_out, seq_len),
#     'mWDN': mWDN(nvars, c_out, seq_len),
#     'mWDNPlus': mWDNPlus(nvars, c_out, seq_len),
#     'TSSequencer': TSSequencer(nvars, c_out, seq_len),
#     'TSSequencerPlus': TSSequencerPlus(nvars, c_out, seq_len),
#     "PatchTST": PatchTST(nvars, c_out, seq_len),
#     "ConvTran": ConvTran(nvars, c_out, seq_len),
#     "ConvTranPlus": ConvTranPlus(nvars, c_out, seq_len),
#     "RNNAttention": RNNAttention(nvars, c_out, seq_len),
#     "LSTMAttention": LSTMAttention(nvars, c_out, seq_len), 
#     "GRUAttention": GRUAttention(nvars, c_out, seq_len),
#     "RNNAttentionPlus": RNNAttentionPlus(nvars, c_out, seq_len), 
#     "LSTMAttentionPlus": LSTMAttentionPlus(nvars, c_out, seq_len),
#     "GRUAttentionPlus": GRUAttentionPlus(nvars, c_out, seq_len),
#     "TransformerRNNPlus": TransformerRNNPlus(nvars, c_out, seq_len), 
#     "TransformerLSTMPlus": TransformerLSTMPlus(nvars, c_out, seq_len),
#     "TransformerGRUPlus": TransformerGRUPlus(nvars, c_out, seq_len), 
#     "Hydra": Hydra(nvars, c_out, seq_len), 
#     "HydraPlus": HydraPlus(nvars, c_out, seq_len),
#     "HydraMultiRocket": HydraMultiRocket(nvars, c_out, seq_len), 
#     "HydraMultiRocketPlus": HydraMultiRocketPlus(nvars, c_out, seq_len),

#     'TabFusionTransformer': TabFusionTransformer(classes, cont_names, c_out),
#     'TSTabFusionTransformer': TSTabFusionTransformer(nvars, c_out, seq_len, classes, cont_names),
#     'TabModel': TabModel(emb_szs, n_cont, c_out),
#     'TabTransformer': TabTransformer(classes, cont_names, c_out),
#     'GatedTabTransformer': GatedTabTransformer(classes, cont_names, c_out),

#     'MiniRocket': MiniRocket(nvars, c_out, seq_len),
#     'MiniRocketPlus': MiniRocketPlus(nvars, c_out, seq_len),
#     'MiniRocketClassifier':MiniRocketClassifier(), # cls.fit(X_train, y_train) cls.score(X_test, y_test)
#     'MiniRocketVotingClassifier': MiniRocketVotingClassifier(n_estimators),
#     'MiniRocketRegressor': MiniRocketRegressor(), # cls.fit(X_train, y_train) cls.predict(X_test, y_test)
#     'MiniRocketVotingRegressor': MiniRocketVotingRegressor(n_estimators),

#     'ROCKET': ROCKET(nvars, seq_len),
#     'RocketClassifier': RocketClassifier(),# cls.fit(X_train, y_train) cls.score(X_test, y_test)
#     'RocketRegressor': RocketRegressor(),# cls.fit(X_train, y_train) cls.predict(X_test, y_test)
    
#     'MultiInputNet': MultiInputNet,
# }

datasets = [
'weight_estimation',
# 'ArticularyWordRecognition',
# 'AtrialFibrillation',
# 'BasicMotions',
# 'Cricket',
# 'EigenWorms',
# 'Epilepsy',
# 'EthanolConcentration',
# 'ERing',
# 'FaceDetection',
# 'FingerMovements',
# 'HandMovementDirection',
# 'Handwriting',
# 'Heartbeat',
# 'Libras',
# 'LSST',
# 'NATOPS',
# 'PenDigits',
# 'Phoneme',
# 'RacketSports',
# 'SelfRegulationSCP1',
# 'SelfRegulationSCP2',
# 'StandWalkJump',
# 'UWaveGestureLibrary',
# 'ACSF1',
# 'Adiac',
# 'AllGestureWiimoteX',
# 'AllGestureWiimoteY',
# 'AllGestureWiimoteZ',
# 'ArrowHead',
# 'Beef',
# 'BeetleFly',
# 'BirdChicken',
# 'BME',
# 'Car',
# 'CBF',
# 'Chinatown',
# 'ChlorineConcentration',
# 'CinCECGTorso',
# 'Coffee',
# 'Computers',
# 'CricketX',
# 'CricketY',
# 'CricketZ',
# 'Crop',
# 'DiatomSizeReduction',
# 'DistalPhalanxOutlineAgeGroup',
# 'DistalPhalanxOutlineCorrect',
# 'DistalPhalanxTW',
# 'DodgerLoopDay',
# 'DodgerLoopGame',
# 'DodgerLoopWeekend',
# 'Earthquakes',
# 'ECG200',
# 'ECG5000',
# 'ECGFiveDays',
# 'ElectricDevices',
# 'EOGHorizontalSignal',
# 'EOGVerticalSignal',
# 'EthanolLevel',
# 'FaceAll',
# 'FaceFour',
# 'FacesUCR',
# 'FiftyWords',
# 'Fish',
# 'FordA',
# 'FordB',
# 'FreezerRegularTrain',
# 'FreezerSmallTrain',
# 'Fungi',
# 'GestureMidAirD1',
# 'GestureMidAirD2',
# 'GestureMidAirD3',
# 'GesturePebbleZ1',
# 'GesturePebbleZ2',
# 'GunPoint',
# 'GunPointAgeSpan',
# 'GunPointMaleVersusFemale',
# 'GunPointOldVersusYoung',
# 'Ham',
# 'HandOutlines',
# 'Haptics',
# 'Herring',
# 'HouseTwenty',
# 'InlineSkate',
# 'InsectEPGRegularTrain',
# 'InsectEPGSmallTrain',
# 'InsectWingbeatSound',
# 'ItalyPowerDemand',
# 'LargeKitchenAppliances',
# 'Lightning2',
# 'Lightning7',
# 'Mallat',
# 'Meat',
# 'MedicalImages',
# 'MelbournePedestrian',
# 'MiddlePhalanxOutlineAgeGroup',
# 'MiddlePhalanxOutlineCorrect',
# 'MiddlePhalanxTW',
# 'MixedShapesRegularTrain',
# 'MixedShapesSmallTrain',
# 'MoteStrain',
# 'OliveOil',
# 'OSULeaf',
# 'PhalangesOutlinesCorrect',
# 'Phoneme',
# 'PickupGestureWiimoteZ',
# 'PigAirwayPressure',
# 'PigArtPressure',
# 'PigCVP',
# 'PLAID',
# 'Plane',
# 'PowerCons',
# 'ProximalPhalanxOutlineAgeGroup',
# 'ProximalPhalanxOutlineCorrect',
# 'ProximalPhalanxTW',
# 'RefrigerationDevices',
# 'Rock',
# 'ScreenType',
# 'SemgHandGenderCh2',
# 'SemgHandMovementCh2',
# 'SemgHandSubjectCh2',
# 'ShakeGestureWiimoteZ',
# 'ShapeletSim',
# 'ShapesAll',
# 'SmallKitchenAppliances',
# 'SmoothSubspace',
# 'SonyAIBORobotSurface1',
# 'SonyAIBORobotSurface2',
# 'StarLightCurves',
# 'Strawberry',
# 'SwedishLeaf',
# 'Symbols',
# 'SyntheticControl',
# 'ToeSegmentation1',
# 'ToeSegmentation2',
# 'Trace',
# 'TwoLeadECG',
# 'TwoPatterns',
# 'UMD',
# 'UWaveGestureLibraryAll',
# 'UWaveGestureLibraryX',
# 'UWaveGestureLibraryY',
# 'UWaveGestureLibraryZ',
# 'Wafer',
# 'Wine',
# 'WordSynonyms',
# 'Worms',
# 'WormsTwoClass',
# 'Yoga',
]

def get_data_list(data_array, data_y, seq_len, step):
    data_x_list = []
    data_y_list = []
    index = 0
    while index + seq_len <= data_array.shape[1]: 
        data_x = data_array[:, index : index + seq_len]
        index += step
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
    interval_length = 2
    seq_len = 144
    step = 10
    for file_name in file_list:
        name, extension = os.path.splitext(file_name)
        file_path = os.path.join(data_dir, file_name)
        if (extension == '.csv'):
            raw_data = pd.read_csv(file_path)
            weight_gt = str(int(raw_data['weight'].mean()/interval_length)*interval_length)

            raw_data = raw_data.drop(['weight','benchmark','yaw_rate','steering_angle'], axis=1)
            
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
            data_x_list, data_y_list = get_data_list(data_array, weight_gt, seq_len, step)
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


# dsid = 'WalkingSittingStanding'
csv_file_path = 'result.csv'
for dsid in datasets:
    print(f'dataset is {dsid}')
    if(dsid == 'weight_estimation'):
        data_dir = '/home/jianhao_dong/NN_weight_estimator/data/weight_estimation_tsai'
        # data_dir = '/home/jianhao_dong/NN_weight_estimator/data/weight_tsai'
        x, y, splits = load_data(data_dir)
    else:
        x, y, splits = get_UCR_data(dsid, return_split=False)
    # tfms  = [None, [Categorize()]]
    # dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=True)
    # dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 64], batch_tfms=[TSStandardize()], num_workers=2)

    tfms = [None, Categorize()]
    batch_tfms = TSStandardize()
    dls = get_ts_dls(x, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=256)
    dls.show_batch(sharey=True)
    all_arch_names =  [
                    # 'FCN', 'FCNPlus', 'InceptionTime', 'InceptionTimePlus', 'InCoordTime', 'XCoordTime', 
                    # 'InceptionTimePlus17x17', 'InceptionTimePlus32x32', 'InceptionTimePlus47x47', 'InceptionTimePlus62x62',  'MultiInceptionTimePlus', 
                    # 'MultiRocket',
                    'MultiRocketPlus', 'InceptionRocketPlus', 
                    'MLP', 'OmniScaleCNN', 'RNN', 'LSTM', 'GRU', 
                    'RNNPlus', 'LSTMPlus', 'GRUPlus', 'RNN_FCN', 'LSTM_FCN', 'GRU_FCN', 'MRNN_FCN', 'MLSTM_FCN', 'MGRU_FCN', 
                    'RNN_FCNPlus', 'LSTM_FCNPlus', 'GRU_FCNPlus', 'MRNN_FCNPlus', 'MLSTM_FCNPlus', 'MGRU_FCNPlus', 
                    'ResCNN', 'ResNet', 'ResNetPlus', 'TCN', 
                    'XCM', 'XCMPlus', 
                    'xresnet1d18', 'xresnet1d34', 'xresnet1d50', 'xresnet1d101', 'xresnet1d152', 'xresnet1d18_deep', 'xresnet1d34_deep', 'xresnet1d50_deep', 
                    'xresnet1d18_deeper', 'xresnet1d34_deeper', 'xresnet1d50_deeper', 
                    
                    'xresnet1d18plus', 'xresnet1d34plus', 
                    'xresnet1d50plus', 'xresnet1d101plus', 
                    'xresnet1d152plus', 'xresnet1d18_deepplus', 'xresnet1d34_deepplus', 'xresnet1d50_deepplus', 
                    'xresnet1d18_deeperplus', 'xresnet1d34_deeperplus', 'xresnet1d50_deeperplus', 'XceptionTime', 'XceptionTimePlus', 'mWDN', 'mWDNPlus',
                    'TSSequencer', 
                    'TSSequencerPlus',
                    "ConvTran", "ConvTranPlus",
                    "RNNAttention", "LSTMAttention", "GRUAttention", 
                    "TransformerRNNPlus", "TransformerLSTMPlus", "TransformerGRUPlus", "Hydra", "HydraPlus", "HydraMultiRocket", "HydraMultiRocketPlus",

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
    unique_labels, label_counts = np.unique(y, return_counts=True)
    print(f'unique labels:{unique_labels},label_counts:{label_counts}')
    re = pd.DataFrame()
    for model_name in all_arch_names:
        try:
            print(f'current model is {model_name}')
            model = model_mapping[model_name]
            learn = ts_learner(dls, model, c_in=dls.vars, c_out=dls.c, seq_len=dls.len, metrics=accuracy)
            print(f'feature_num:{dls.vars}, class num:{dls.c}')
            # learn.lr_find()
            learn.fit_one_cycle(5, lr_max=1e-3)
        except Exception:
            continue
        final_record = learn.final_record
        # learn.show_results()
        # learn.show_probas()
        # learn.plot_confusion_matrix()
        
        df = pd.DataFrame([{
                        'dataset':dsid,
                        'feature_num': dls.vars,
                        'class_num': dls.c,
                        'seq_len': dls.len,
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