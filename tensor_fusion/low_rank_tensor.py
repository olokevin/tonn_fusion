import tltorch
import torch
import torch.nn as nn
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.normal import Normal
from .distribution import LogUniform
from torch.nn.init import xavier_normal_
import numpy as np

class LowRankTensor(nn.Module):

    def __init__(self, in_features, out_features, min_dim=4, prior_type='log_uniform', eta=None, device=None, dtype=None):

        super().__init__()

        if prior_type == 'half_cauchy':
            self.rank_parameter_prior_distribution = HalfCauchy(eta)
        elif prior_type == 'log_uniform':
            self.rank_parameter_prior_distribution = LogUniform(torch.tensor([1e-30], device=device, dtype=dtype), 
                                                                torch.tensor([1e30], device=device, dtype=dtype))

        self.tensorized_shape = tltorch.utils.get_tensorized_shape(in_features, out_features, order = 4, min_dim=min_dim, verbose=True)

class CP(LowRankTensor):

    def __init__(self, in_features, out_features, max_rank, min_dim, prior_type='log_uniform', eta=None, device=None, dtype=None):

        super().__init__(in_features, out_features, min_dim, prior_type, eta, device, dtype)

        self.tensor = tltorch.TensorizedTensor.new(tensorized_shape=self.tensorized_shape,
                                                   rank=max_rank,
                                                   factorization='CP',
                                                   device=device,
                                                   dtype=dtype)

        target_var = 1 / in_features
        factor_std = (target_var / self.tensor.rank) ** (1 / (4 * self.tensor.order))
        for factor in self.tensor.factors:
            nn.init.normal_(factor, 0, factor_std)

        self.rank_parameters = nn.Parameter(torch.rand((max_rank,), device=device, dtype=dtype))

    def get_log_prior(self):

        with torch.no_grad():
            self.rank_parameters[:] = self.rank_parameters.clamp(1e-10)
        
        # self.threshold(self.rank_parameter)
        log_prior = torch.sum(self.rank_parameter_prior_distribution.log_prob(self.rank_parameters))
        
        # 0 mean normal distribution for the factors
        factor_prior_distribution = Normal(0, self.rank_parameters)
        for factor in self.tensor.factors:
            log_prior = log_prior + factor_prior_distribution.log_prob(factor).sum(0).sum()
        
        return log_prior
    
    def estimate_rank(self):
        
        rank = 0
        for factor in self.tensor.factors:
            factor_rank = torch.sum(factor.var(axis=0) > 1e-5)
            rank = max(rank, factor_rank)
        
        return rank

class Tucker(LowRankTensor):

    def __init__(self, in_features, out_features, max_rank, min_dim, prior_type='log_uniform', eta=None, device=None, dtype=None):

        super().__init__(in_features, out_features, min_dim, prior_type, eta, device, dtype)

        self.tensor = tltorch.TensorizedTensor.new(tensorized_shape=self.tensorized_shape,
                                                   rank=max_rank,
                                                   factorization='Tucker',
                                                   device=device,
                                                   dtype=dtype)
        tltorch.tensor_init(self.tensor)

        self.rank_parameters = nn.ParameterList([nn.Parameter(torch.rand((max_rank,), device=device, dtype=dtype)) \
            for _ in range(len(self.tensor.factors))])

    def get_log_prior(self):

        log_prior = 0.0
        for r in self.rank_parameters:
            # clamp rank_param because <=0 is undefined 
            with torch.no_grad():
                r[:] = r.clamp(1e-10)
            log_prior = log_prior + torch.sum(self.rank_parameter_prior_distribution.log_prob(r))
        
        core_prior_distribution = Normal(0, 10.0)
        log_prior = log_prior + torch.sum(core_prior_distribution.log_prob(self.tensor.core))

        for r, factor in zip(self.rank_parameters, self.tensor.factors):
            factor_prior_distribution = Normal(0, r)
            log_prior = log_prior + torch.sum(factor_prior_distribution.log_prob(factor))

        return log_prior

class TT(LowRankTensor):

    def __init__(self, in_features, out_features, max_rank, min_dim, prior_type='log_uniform', eta=None, device=None, dtype=None):

        super().__init__(in_features, out_features, min_dim, prior_type, eta, device, dtype)

        tensorized_dim = [*self.tensorized_shape[0], *self.tensorized_shape[1]]
        self.tensor = tltorch.TTTensor.new(shape=tensorized_dim, rank=max_rank, device=device, dtype=dtype)

        target_var = 1 / in_features
        factor_std = ((target_var / np.prod(self.tensor.rank)) ** (1 / self.tensor.order)) ** 0.5

        for factor in self.tensor.factors:
            nn.init.normal_(factor, 0, factor_std)

        self.rank_parameters = nn.ParameterList([nn.Parameter(torch.rand((max_rank,), device=device, dtype=dtype)) \
            for _ in range(len(tensorized_dim)-1)])

    def get_log_prior(self):

        log_prior = 0.0
        for r in self.rank_parameters:
            # clamp rank_param because <=0 is undefined 
            with torch.no_grad():
                r[:] = r.clamp(1e-10)
            log_prior = log_prior + torch.sum(self.rank_parameter_prior_distribution.log_prob(r))

        
        # 0 mean normal distribution for the factors
        for i in range(len(self.rank_parameters)):
            factor_prior_distribution = Normal(0, self.rank_parameters[i])
            log_prior = log_prior + factor_prior_distribution.log_prob(self.tensor.factors[i]).sum((0,1)).sum()
        
        factor_prior_distribution = Normal(0, self.rank_parameters[-1])
        log_prior = log_prior + factor_prior_distribution.log_prob(self.tensor.factors[-1]).sum((1,2)).sum()

        return log_prior
    
    def estimate_rank(self):

        rank = [1]
        for factor in self.tensor.factors[:-1]:
            rank.append(torch.sum(factor.var((0,1)) > 1e-5))
        
        rank.append(torch.sum(self.tensor.factors[-1].var((1,2)) > 1e-5))
        
        return rank

class TTM(LowRankTensor):

    def __init__(self, in_features, out_features, max_rank, min_dim, prior_type='log_uniform', eta=None, device=None, dtype=None):

        super().__init__(in_features, out_features, min_dim, prior_type, eta, device, dtype)

        self.tensor = tltorch.TensorizedTensor.new(tensorized_shape=self.tensorized_shape,
                                              rank=max_rank,
                                              factorization='blocktt',
                                              dtype=dtype,
                                              device=device)
        # previous init:
        # tltorch.tensor_init(self.tensor)

        # 221128 add init value
        target_var = 1 / in_features
        factor_std = ((target_var / np.prod(self.tensor.rank)) ** (1 / self.tensor.order)) ** 0.5

        for factor in self.tensor.factors:
            nn.init.normal_(factor, 0, factor_std)

        self.rank_parameters = nn.ParameterList([nn.Parameter(torch.rand((max_rank,), device=device, dtype=dtype)) \
            for _ in range(len(self.tensor.factors)-1)])

    def get_log_prior(self):

        log_prior = 0.0
        for r in self.rank_parameters:
            # clamp rank_param because <=0 is undefined 
            with torch.no_grad():
                r[:] = r.clamp(1e-10)
            log_prior = log_prior + torch.sum(self.rank_parameter_prior_distribution.log_prob(r))
    
        # 0 mean normal distribution for the factors
        for i in range(len(self.rank_parameters)):
            factor_prior_distribution = Normal(0, self.rank_parameters[i])
            log_prior = log_prior + torch.sum(factor_prior_distribution.log_prob(self.tensor.factors[i]))
        
        factor_prior_distribution = Normal(0, self.rank_parameters[-1])
        log_prior = log_prior + torch.sum(factor_prior_distribution.log_prob(self.tensor.factors[-1]))

        return log_prior