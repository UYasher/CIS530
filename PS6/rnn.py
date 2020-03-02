from PS6.load_names import n_letters, n_categories
from PS6.char_to_tensor import lineToTensor
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data, hidden):
        combined = torch.cat((input_data, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

#
# n_hidden = 128
# rnn = RNN(n_letters, n_hidden, n_categories)
#
# inputs = lineToTensor('Albert')
# hidden_layer = torch.zeros(1, n_hidden)
#
# output, next_hidden = rnn(inputs[0], hidden_layer)
# print(output)
