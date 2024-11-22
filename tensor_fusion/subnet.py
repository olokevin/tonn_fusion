import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import AdaptiveRankLinear, AdaptiveRankLSTM

class InferenceSubNet(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0, device=None, dtype=None):

        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(in_size, in_size, device=device, dtype=dtype)
        self.linear_2 = nn.Linear(in_size, out_size, device=device, dtype=dtype)

    def forward(self, x):
        output = F.relu(self.linear_1(x))
        output = self.dropout(output)
        output = self.linear_2(output)
        return output

class SubNet(nn.Module):

    def __init__(self, in_size, hidden_size, dropout=0.2, device=None, dtype=None):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super().__init__()

        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size, device=device, dtype=dtype)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.linear_3 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''

        output = F.relu(self.linear_1(x))
        output = self.drop(output)
        output = F.relu(self.linear_2(output))
        output = self.drop(output)
        output = F.relu(self.linear_3(output))
        output = self.drop(output)

        return output

class TextSubNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False, device=None, dtype=None):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size, device=device, dtype=dtype)
        self.linear_2 = nn.Linear(out_size, out_size, device=device, dtype=dtype)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        output = F.relu(self.linear_1(final_states[0].squeeze()))
        output = self.dropout(output)
        output = F.relu(self.linear_2(output))
        output = self.dropout(output)
        return output

class AdaptiveRankTextSubNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, dropout=0.2, bias=True,
                 max_rank=20, tensor_type='TT', prior_type='log_uniform', eta=None,
                 device=None, dtype=None):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            out_size: output dimension
            dropout: dropout probability
            max_rank: maximum rank for LSTM's weight tensor
            tensor_type: LSTM's weight tensor type 'CP', 'Tucker', 'TT' or 'TTM'
            prior_type: prior for the rank parameter 'log_uniform' or 'half_cauchy'
            eta: hyperparameter for the 'half_cauchy' distribution
            device:
            dtype:
            
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()

        self.rnn = AdaptiveRankLSTM(in_size, hidden_size, max_rank, bias,
                                    tensor_type, prior_type, eta, device, dtype)
        
        self.dropout = nn.Dropout(dropout)
        
        self.linear_1 = AdaptiveRankLinear(hidden_size, out_size, max_rank, bias,
                                           tensor_type, prior_type, eta, device, dtype)
        self.linear_2 = AdaptiveRankLinear(out_size, out_size, max_rank, bias,
                                           tensor_type, prior_type, eta, device, dtype)
        

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        output = F.relu(self.linear_1(final_states[0].squeeze()))
        output = self.dropout(output)
        output = F.relu(self.linear_2(output))
        output = self.dropout(output)
        return output

    def get_log_prior(self):

        return self.rnn.get_log_prior() + self.linear_1.get_log_prior() + self.linear_2.get_log_prior()

