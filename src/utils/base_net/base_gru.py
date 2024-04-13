from torch import nn

class BaseGRU(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, num_layers):
        super(BaseGRU, self).__init__()

        self.gru = nn.GRU(inp_dim, hidden_dim, num_layers, batch_first = True)
        self.dropout = nn.Dropout(0.5)
        self.reg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )
        

    def forward(self, sequences):
        # gru_y, h = self.gru(sequences)
        # gru_y = gru_y[:,-1,:]
        # gru_y = self.dropout(gru_y)
        # y = self.reg(gru_y)
        # return y
        seq_len = sequences.shape[1]
        gru_out, self.hidden = self.gru(
            sequences.view(len(sequences), seq_len, -1)
        )
        y_pred = self.reg(gru_out.view(len(sequences), seq_len, -1))
        return y_pred