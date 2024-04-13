import torch

batch_size = 1
seq_len = 5
input_size = 4
hidden_size =2
num_layers = 3
#其他参数
#batch_first=True 维度从(SeqLen*Batch*input_size)变为（Batch*SeqLen*input_size）
cell = torch.nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)

inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)

out, hidden = cell(inputs, hidden)

print("Output size: ", out.shape)
print("Output: ", out)
print("Hidden size: ", hidden.shape)
print("Hidden: ", hidden)