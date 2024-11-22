import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.normal import Normal

import numpy as np
from .distribution import LogUniform
from . import low_rank_tensor as LowRankTensor

from .tensor_times_matrix import tensor_times_matrix_fwd

class TensorFusion(nn.Module):

    def __init__(self, input_sizes, output_size, dropout=0.0, bias=True, device=None, dtype=None):

        super().__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        
        # initialize weight tensor
        tensorized_shape = input_sizes + (output_size,)
        self.weight_tensor = nn.Parameter(torch.empty(tensorized_shape, device=device, dtype=dtype))
        nn.init.xavier_normal_(self.weight_tensor)

        # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((output_size,), device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, inputs):

        fusion_tensor = inputs[0]
        for x in inputs[1:]:
            fusion_tensor = torch.einsum('n...,na->n...a', fusion_tensor, x)
        
        fusion_tensor = self.dropout(fusion_tensor)

        output = torch.einsum('n...,...o->no', fusion_tensor, self.weight_tensor)

        if self.bias is not None:
            output = output + self.bias

        output = F.relu(output)
        output = self.dropout(output)

        return output

class LowRankFusion(nn.Module):

    def __init__(self, input_sizes, output_size, rank, dropout=0.0, bias=True, device=None, dtype=None):

        super().__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.rank = rank
        self.dropout = nn.Dropout(dropout)

        # initialize weight tensor factors
        factors = [nn.Parameter(torch.empty((input_size, rank), device=device, dtype=dtype)) \
            for input_size in input_sizes]
        factors = factors + [nn.Parameter(torch.empty((output_size, rank), device=device, dtype=dtype))]
        
        for factor in factors:
            nn.init.xavier_normal_(factor)

        self.weight_tensor_factors = nn.ParameterList(factors)

        # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((output_size,), device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, inputs):
        
        # tensorized forward propagation
        output = 1.0
        for x, factor in zip(inputs, self.weight_tensor_factors[:-1]):
            output = output * (x @ factor)

        output = output @ self.weight_tensor_factors[-1].T

        if self.bias is not None:
            output = output + self.bias

        output = F.relu(output)
        output = self.dropout(output)

        return output

class AdaptiveRankFusion(nn.Module):

    def __init__(self, input_sizes, output_size, dropout=0.0, bias=True,
                 max_rank=20, prior_type='log_uniform', eta=None, 
                 device=None, dtype=None):

        super().__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.max_rank = max_rank
        self.dropout = nn.Dropout(dropout)

        # initialize weight tensor factors
        factors = [nn.Parameter(torch.empty((input_size, max_rank), device=device, dtype=dtype)) \
            for input_size in input_sizes]
        factors = factors + [nn.Parameter(torch.empty((output_size, max_rank), device=device, dtype=dtype))]
        
        target_var = 1 / np.prod(input_sizes)
        factor_std = (target_var / max_rank) ** (1 / (4 * 4))
        for factor in factors:
            nn.init.normal_(factor, 0, factor_std)

        self.weight_tensor_factors = nn.ParameterList(factors)

        self.rank_parameters = nn.Parameter(torch.rand((max_rank,), device=device, dtype=dtype))

        if prior_type == 'half_cauchy':
            self.rank_parameter_prior_distribution = HalfCauchy(eta)
        elif prior_type == 'log_uniform':
            self.rank_parameter_prior_distribution = LogUniform(torch.tensor([1e-30], device=device, dtype=dtype), 
                                                                torch.tensor([1e30], device=device, dtype=dtype))

        # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((output_size,), device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, inputs):
        
        # tensorized forward propagation
        output = 1.0
        for x, factor in zip(inputs, self.weight_tensor_factors[:-1]):
            output = output * (x @ factor)

        output = output @ self.weight_tensor_factors[-1].T

        if self.bias is not None:
            output = output + self.bias

        output = F.relu(output)
        output = self.dropout(output)

        return output

    def get_log_prior(self):
        
        # clamp rank_param because <=0 is undefined 
        with torch.no_grad():
            self.rank_parameters[:] = self.rank_parameters.clamp(1e-10)
        
        # self.threshold(self.rank_parameter)
        log_prior = torch.sum(self.rank_parameter_prior_distribution.log_prob(self.rank_parameters))
        
        # 0 mean normal distribution for the factors
        factor_prior_distribution = Normal(0, self.rank_parameters)
        for factor in self.weight_tensor_factors:
            log_prior = log_prior + factor_prior_distribution.log_prob(factor).sum(0).sum()
        
        return log_prior
    
    def estimate_rank(self):
        
        rank = 0
        for factor in self.weight_tensor_factors:
            factor_rank = torch.sum(factor.var(axis=0) > 1e-5)
            rank = max(rank, factor_rank)
        
        return rank

class AdaptiveRankLinear(nn.Module):

    def __init__(self, in_features, out_features, max_rank, min_dim=4, bias=True, tensor_type='TT', prior_type='log_uniform',
                 eta=None, device=None, dtype=None):
        '''
        Args:
            in_features: input dimension
            out_features: output dimension
            max_rank: maximum rank for LSTM's weight tensor
            min_dim:  smallest acceptable integer value for factorization factors
            bias: add bias?
            tensor_type: LSTM's weight tensor type 'CP', 'Tucker', 'TT' or 'TTM'
            prior_type: prior for the rank parameter 'log_uniform' or 'half_cauchy'
            eta: hyperparameter for the 'half_cauchy' distribution
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()

        self.weight_tensor = getattr(LowRankTensor, tensor_type)(in_features, out_features, max_rank, min_dim, prior_type, eta, device, dtype)
        
       # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):

        output = tensor_times_matrix_fwd(self.weight_tensor.tensor, x.T)

        if self.bias is not None:
            output = output + self.bias
            
        return output

    def get_log_prior(self):

        return self.weight_tensor.get_log_prior()

    def estimate_rank(self):

        return self.weight_tensor.estimate_rank()

class AdaptiveRankLSTM(nn.Module):
    '''
    no frills batch first LSTM implementation
    '''
    def __init__(self, input_size, hidden_size, max_rank, bias=True,
                 tensor_type='TT', prior_type='log_uniform', eta=None,
                 device=None, dtype=None):
        '''
        args:
            input_size: input dimension size
            hidden_size: hidden dimension size
            max_rank: maximum rank for LSTM's weight tensor
            tensor_type: LSTM's weight tensor type 'CP', 'Tucker', 'TT' or 'TTM'
            prior_type: prior for the rank parameter 'log_uniform' or 'half_cauchy'
            eta: hyperparameter for the 'half_cauchy' distribution
            device:
            dtype:
        '''
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih_min_dim = 3
        self.hh_min_dim = 2
        
        self.layer_ih = AdaptiveRankLinear(input_size, hidden_size*4,  max_rank, self.ih_min_dim, bias, 
                                           tensor_type, prior_type, eta, device, dtype)
        self.layer_hh = AdaptiveRankLinear(hidden_size, hidden_size*4, max_rank, self.hh_min_dim, bias,
                                           tensor_type, prior_type, eta, device, dtype)
        
    def forward(self, x):

        # LSTM forward propagation
        output = []
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        c = torch.zeros((batch_size, self.hidden_size), device=x.device, dtype=x.dtype)
        h = torch.zeros((batch_size, self.hidden_size), device=x.device, dtype=x.dtype)
        for seq in range(seq_length):
            ih = self.layer_ih(x[:,seq,:])
            hh = self.layer_hh(h)
            i, f, g, o = torch.split(ih + hh, self.hidden_size, 1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            output.append(h.unsqueeze(1))
            
        output = torch.cat(output, dim=1)
        
        return output, (h, c)

    def get_log_prior(self):

        return self.layer_ih.get_log_prior() + self.layer_hh.get_log_prior()



class AdaptiveRankLSTM8(nn.Module):
    '''
    no frills batch first LSTM implementation
    '''
    def __init__(self, input_size, hidden_size, max_rank, bias=True,
                 tensor_type='TT', prior_type='log_uniform', eta=None,
                 device=None, dtype=None):
        '''
        args:
            input_size: input dimension size
            hidden_size: hidden dimension size
            max_rank: maximum rank for LSTM's weight tensor
            tensor_type: LSTM's weight tensor type 'CP', 'Tucker', 'TT' or 'TTM'
            prior_type: prior for the rank parameter 'log_uniform' or 'half_cauchy'
            eta: hyperparameter for the 'half_cauchy' distribution
            device:
            dtype:
        '''
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.layer_ih_I = AdaptiveRankLinear(input_size, hidden_size, max_rank, bias, 
                                           tensor_type, prior_type, eta, device, dtype)
        self.layer_ih_F = AdaptiveRankLinear(input_size, hidden_size, max_rank, bias, 
                                           tensor_type, prior_type, eta, device, dtype)
        self.layer_ih_G = AdaptiveRankLinear(input_size, hidden_size, max_rank, bias, 
                                           tensor_type, prior_type, eta, device, dtype)
        self.layer_ih_O = AdaptiveRankLinear(input_size, hidden_size, max_rank, bias, 
                                           tensor_type, prior_type, eta, device, dtype)                               
        
        
        self.layer_hh_I = AdaptiveRankLinear(hidden_size, hidden_size, max_rank, bias,
                                           tensor_type, prior_type, eta, device, dtype)
        self.layer_hh_F = AdaptiveRankLinear(hidden_size, hidden_size, max_rank, bias,
                                           tensor_type, prior_type, eta, device, dtype)
        self.layer_hh_G = AdaptiveRankLinear(hidden_size, hidden_size, max_rank, bias,
                                           tensor_type, prior_type, eta, device, dtype)
        self.layer_hh_O = AdaptiveRankLinear(hidden_size, hidden_size, max_rank, bias,
                                           tensor_type, prior_type, eta, device, dtype)
        
    def forward(self, x):

        # LSTM forward propagation
        output = []
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        c = torch.zeros((batch_size, self.hidden_size), device=x.device, dtype=x.dtype)
        h = torch.zeros((batch_size, self.hidden_size), device=x.device, dtype=x.dtype)
        for seq in range(seq_length):
            ih_I = self.layer_ih_I(x[:,seq,:])
            ih_F = self.layer_ih_F(x[:,seq,:])
            ih_G = self.layer_ih_G(x[:,seq,:])
            ih_O = self.layer_ih_O(x[:,seq,:])
            hh_I = self.layer_hh_I(h)
            hh_F = self.layer_hh_F(h)
            hh_G = self.layer_hh_G(h)
            hh_O = self.layer_hh_O(h)

            i, f, g, o = ih_I + hh_I,  ih_F + hh_F, ih_G + hh_G, ih_O + hh_O
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            output.append(h.unsqueeze(1))
            
        output = torch.cat(output, dim=1)
        
        return output, (h, c)

    def get_log_prior(self):

        return self.layer_ih.get_log_prior() + self.layer_hh.get_log_prior()