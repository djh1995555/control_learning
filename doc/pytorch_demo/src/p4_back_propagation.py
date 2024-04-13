import torch
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0]) # w的初值为1.0
w.requires_grad = True # 需要计算梯度

def forward(x):
    return x*w  # w是一个Tensor


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l =loss(x,y) # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward() #  backward,compute grad for Tensor whose requires_grad set to True
        print('\tgrad:', x, y, w.grad.item())
        # 权重更新时，注意grad也是一个tensor
        # 如果不用.data，直接加的化会导致tensor相加，形成一个计算图，占用极大内存
        w.data = w.data - 0.01 * w.grad.data
        # 每次计算完需要对梯度进行清零，否则会累加
        w.grad.data.zero_()

    print('progress:', epoch, l.item()) # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print("predict (after training)", 4, w.data)