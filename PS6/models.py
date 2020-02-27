import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

'''
Please add default values for all the parameters of __init__.
'''
class CharRNNClassify(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNNClassify, self).__init__()

        # self.hidden_size = hidden_size
        #
        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden=None):
        # combined = torch.cat((input, hidden), 1)
        # hidden = self.i2h(combined)
        # output = self.i2o(combined)
        # output = self.softmax(output)
        lstm_out, _ = self.lstm(input)
        output = self.hidden_layer(lstm_out)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0")
x = torch.randn(600, 50, device=device, dtype=dtype)
