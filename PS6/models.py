import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

'''
Please add default values for all the parameters of __init__.
'''
class CharRNNClassify(nn.Module):

    def __init__(self):
        super(CharRNNClassify, self).__init__()

        self.input_size = 57
        self.hidden_size = 20
        self.output_size = 9

        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, inputs, hidden=None):
        lstm_out, hidden = self.lstm(inputs.view(inputs.size(1), 1, -1))
        output = self.linear(lstm_out[-1, :, :])
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.randn(1, self.hidden_size)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dtype = torch.float
# device = torch.device("cuda:0")
# x = torch.randn(600, 50, device=device, dtype=dtype)
