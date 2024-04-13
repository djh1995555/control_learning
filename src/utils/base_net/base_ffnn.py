from torch import nn

class BaseFFNN(nn.Module):
    def __init__(self, input_dim, dim_list, output_dim):
        super(BaseFFNN, self).__init__()
        dim_list.insert(0, input_dim)
        self._model = nn.Sequential()
        for i in range(len(dim_list) - 1):
            self._model.add_module("fc{}".format(i), nn.Linear(dim_list[i], dim_list[i + 1]))
            self._model.add_module("relu{}".format(i), nn.ReLU())

        self._model.add_module("output", nn.Linear(dim_list[-1], output_dim))

    def forward(self, x):
        y = self._model(x)
        return y