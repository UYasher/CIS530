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
        self.hidden_size = 9

        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, inputs, hidden=None):
        lstm_out, hidden = self.lstm(inputs.view(inputs.size(1), 1, -1))
        output = self.softmax(lstm_out[-1, :, :])
        return output, hidden

    def init_hidden(self):
        return torch.randn(1, self.hidden_size)

# dtype = torch.float
# device = torch.device("cuda:0")
# x = torch.randn(600, 50, device=device, dtype=dtype)
