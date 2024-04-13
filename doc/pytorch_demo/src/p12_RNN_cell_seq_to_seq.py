import torch

input_size = 4
hidden_size = 3
batch_size = 1

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
#构造独热向量，此时向量维度为(SeqLen*InputSize)
x_one_hot = [one_hot_lookup[x] for x in x_data]
#view(-1……)保留原始SeqLen，并添加batch_size,input_size两个维度
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
#将labels转换为（SeqLen*1）的维度
labels = torch.LongTensor(y_data).view(-1, 1)

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnncell = torch.nn.RNNCell(input_size = self.input_size,
                                        hidden_size = self.hidden_size)

    def forward(self, input, hidden):
        # RNNCell input = (batchsize*inputsize)
        # RNNCell hidden = (batchsize*hiddensize)
        hidden = self.rnncell(input, hidden)
        return hidden

    #初始化零向量作为h0，只有此处用到batch_size
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)

net = Model(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

total_epoch = 50
for epoch in range(total_epoch):
    #损失及梯度置0，创建前置条件h0
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()

    print("Predicted string: ",end="")
    #inputs=（seqLen*batchsize*input_size） labels = (seqLen*1)
    #input是按序列取的inputs元素（batchsize*inputsize）
    #label是按序列去的labels元素（1）
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        # 序列的每一项损失都需要累加，因为这个loss是一整个序列的loss
        # 而且不能用item()，因为需要把loss加到计算图
        loss += criterion(hidden, label)
        #多分类取最大
        _, idx = hidden.max(dim=1)
        print(idx2char_2[idx.item()], end='')

    loss.backward()
    optimizer.step()

    print(", Epoch [%d/%d] loss = %.4f" % (total_epoch, epoch+1, loss.item()))