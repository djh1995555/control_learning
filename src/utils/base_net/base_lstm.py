from torch import nn

class BaseLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, mid_layers):
        super(BaseLSTM, self).__init__()
        self.n_hidden = hidden_dim
        self.lstm = nn.LSTM(inp_dim, hidden_dim, mid_layers, batch_first = True, dropout=0.5)
        self.reg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, sequences):
        # lstm_y, (hn,cn) = self.lstm(sequences)
        # lstm_y = lstm_y[:,-1,:]
        # y = self.reg(lstm_y)
        # return y
        seq_len = sequences.shape[1]
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), seq_len, -1)
        )
        y_pred = self.reg(lstm_out.view(len(sequences), seq_len, -1))
        return y_pred