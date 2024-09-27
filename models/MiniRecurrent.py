import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Dave2v1 with Recurrent operator
class MiniRNN(nn.Module):
    def __init__(self, input_shape=(100, 100), rnn_input_size=10, rnn_hidden_size=4, nlayers=4, dropout=0.5, tie_weights=False):
        super(MiniRNN, self).__init__()
        self.input_shape = input_shape
        self.bn1 = nn.BatchNorm2d(3, eps=0.001, momentum=0.99, track_running_stats=False)
        self.drop = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)
        # torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
        self.rnn = nn.RNN(10, rnn_hidden_size, nlayers, nonlinearity="tanh", dropout=dropout, batch_first=True)
        size = np.prod(nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)(
            torch.zeros(1, 3, *self.input_shape)).shape)

        self.lin1 = nn.Linear(in_features=size, out_features=100, bias=True)
        self.lin2 = nn.Linear(in_features=100, out_features=50, bias=True)
        self.lin3 = nn.Linear(in_features=50, out_features=10, bias=True)
        # self.lin4 = nn.Linear(in_features=10, out_features=2, bias=True)
        self.lin4 = nn.Linear(in_features=4, out_features=1, bias=True)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x, h):
        x = self.bn1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = x.flatten(1)
        # print(x.shape)
        x = self.drop(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        # print("Before RNN op:", x.shape, h.shape)
        x, hidden = self.rnn(x, h)
        # print(x.shape, hidden.shape)
        x = F.relu(x)
        x = self.lin4(x)
        x = torch.tanh(x)
        # print(x.shape)
        # x = 2 * torch.atan(x)
        return x, hidden

# https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, input_shape, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.input_shape = input_shape
        self.drop = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)
        ninp = np.prod(nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)(
            torch.zeros(1, 3, *self.input_shape)).shape)
        self.emb = nn.Embedding(ntoken, ninp)
        self.encoder = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5) #, self.emb)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError as e:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""") from e
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        print(emb.shape)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


if __name__ == "__main__":
    # torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
    # rnn = nn.RNN(10, 20, 2)
    # input = torch.randn(5, 3, 10)
    # h0 = torch.randn(2, 3, 20)
    # out, hn = rnn(input, h0)
    # print(out.shape, hn.shape)

    input_shape = (2560, 720)
    # m = RNNModel("RNN_TANH", 4, input_shape, 1, 1)
    m = MiniRNN(input_shape=input_shape, rnn_input_size=10, rnn_hidden_size=4, nlayers=1, dropout=0.5, tie_weights=False)
    print(m)
    img = torch.rand((1, 3, 2560, 720))
    h0 = torch.randn((1,4))
    out, hn = m(img, h0)
    print(out, hn)