import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size =2

cell = torch.nn.RNNCell(input_size = input_size, hidden_size = hidden_size)
#维度最重要
dataset = torch.randn(seq_len,batch_size,input_size)
#初始化时设为零向量
hidden = torch.zeros(batch_size, hidden_size)

for idx,input in enumerate(dataset):
    print('=' * 20,idx,'=' * 20)
    print('Input size:', input.shape)

    hidden = cell(input, hidden)

    print('outputs size: ', hidden.shape)
    print(hidden)