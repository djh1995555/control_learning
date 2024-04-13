import torch

input_size = 4
hidden_size = 3
batch_size = 1
num_layers = 1
seq_len = 5
#构建输入输出字典
idx2char_1 = ['e', 'h', 'l', 'o']
idx2char_2 = ['h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [2, 0, 1, 2, 1]
# y_data = [3, 1, 2, 2, 3]
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
#labels（seqLen*batchSize,1）为了之后进行矩阵运算，计算交叉熵
labels = torch.LongTensor(y_data)

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.batch_size = batch_size #构造H0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size = self.input_size,
                                hidden_size = self.hidden_size,
                                num_layers=num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size)
        out, _ = self.rnn(input, hidden)
        #reshape成（SeqLen*batchsize,hiddensize）便于在进行交叉熵计算时可以以矩阵进行。
        return out.view(-1, self.hidden_size)

net = Model(input_size, hidden_size, batch_size, num_layers)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

#RNN中的输入（SeqLen*batchsize*inputsize）
#RNN中的输出（SeqLen*batchsize*hiddensize）
#labels维度 hiddensize*1
total_epoch = 50
for epoch in range(total_epoch):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted string: ',''.join([idx2char_2[x] for x in idx]), end = '')
    print(", Epoch [%d/%d] loss = %.3f" % (total_epoch, epoch+1, loss.item()))