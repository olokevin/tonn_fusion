from ctypes import Union
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .low_rank_tensors import CP,TensorTrain,TensorTrainMatrix,Tucker
from .emb_utils import get_cum_prod,tensorized_lookup
import tensorly as tl
from . import low_rank_tensors
from .Q_back_tensor import ttm_times_mat


# from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize

#---------------------Define Scale Layer-------------------------------------------------------
def quant_func(x,bit=8):
    max_q = 2.0**(bit)-1.0
    min_q = -2.0**(bit)

    return torch.clip(x.to(int).to(x.device),min=min_q,max=max_q)


class scale(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, scale, bit, half = False):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        if half:
            max_q = 2.0**(bit)-1.0
            min_q = 0
            # quant = lambda x : fixed_point_quantize(x, wl=bit+1, fl=0, rounding="nearest")
            quant = lambda x : quant_func(x, bit=bit+1)
            # quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="stochastic")

        else:
            max_q = 2.0**(bit-1)-1.0
            min_q = -2.0**(bit-1)
            # quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")
            quant = lambda x : quant_func(x, bit=bit)
            # quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="stochastic")


        ctx.save_for_backward(input, scale)
        ctx.quant = quant
        ctx.input_div_scale = input/scale
        ctx.q_input = quant(ctx.input_div_scale)
        ctx.min_q = torch.tensor(min_q)
        ctx.max_q = torch.tensor(max_q)
        return scale * ctx.q_input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, scale= ctx.saved_tensors
        grad_input = grad_output*torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), 1.0, 0.0)
        
        grad_scale = (torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), ctx.q_input - ctx.input_div_scale, ctx.input_div_scale))

        # print(grad_scale.device)
        # print(grad_output.device)

        # print(torch.clamp(grad_scale, min = ctx.min_q, max = ctx.max_q))

        grad_scale = grad_output*torch.clamp(grad_scale, min = ctx.min_q.to(grad_scale.device), max = ctx.max_q.to(grad_scale.device))

        return grad_input, grad_scale, None, None, None




class ScaleLayer(nn.Module):

   def __init__(self, scale=2**(-5), bit = 8, half = True):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([scale]))

       self.bit = bit
       self.half = half

    #    max_q = 2.0**(bit-1)-1.0
    #    min_q = -2.0**(bit-1)
    #    quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

    #    self.quant = quant
    #    self.min_q = min_q
    #    self.max_q = max_q

   def forward(self, input):
       return scale.apply(input,self.scale,self.bit,self.half)


class Quantized_Linear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                eta = None,
                device=None,
                dtype=None,
                bit = 8,
                scale_w = 2**(-5),
                scale_b = 2**(-5)
    ):

        super(Quantized_Linear,self).__init__(in_features,out_features,bias,device,dtype)

        self.in_features = in_features
        self.out_features = out_features

        self.bit = bit

        # self.max_q = 2.0**(bit-1)-1.0
        # self.min_q = -2.0**(bit-1)
        # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))
       

    def forward(self, input):
        self.weight = scale.apply(self.weight,self.scale_w,self.bit)
        self.bias = scale.apply(self.bias,self.scale_b,self.bit)
        
        return F.linear(input,self.weight,self.bias)



class Q_TensorizedLinear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
                bit_w = 8,
                bit_b = 8,
                scale_w = 2**(-5),
                scale_b = 2**(-5),
                Q_back = False,
                recur = False,
    ):

        super(Q_TensorizedLinear,self).__init__(in_features,out_features,bias,device,dtype)

        self.Q_back = Q_back
        self.recur = recur

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        self.bit_w = bit_w
        self.bit_b = bit_b

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        # self.max_q = 2.0**(bit-1)-1.0
        # self.min_q = -2.0**(bit-1)
        # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))


    def forward(self, input, rank_update=True):
        

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
        Q_factors = []        
        for U in self.tensor.factors:
            Q_factors.append(scale.apply(U,self.scale_w,self.bit_w, False))
        if self.Q_back:
            quant = lambda x : fixed_point_quantize(x, wl=8, fl=6, rounding="nearest")
            Q_factors = [torch.swapaxes(U,0,-1) for U in Q_factors[::-1]]
            output = ttm_times_mat.apply(quant,input.T, *Q_factors)
        elif self.recur:
            Q = lambda x: scale.apply(x,self.scale_w,self.bit_w, False)
            output = input @ self.tensor.full_from_factors(Q_factors,quant=Q).reshape([self.out_features,self.in_features]).T
        else:
            output = input @ self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features]).T

        if self.bias is not None:
            Q_bias = (scale.apply(self.bias,self.scale_b,self.bit_b, False))
            output = scale.apply(output,self.scale_b,self.bit_b, False) + Q_bias

        ### Code for test purpose only ##############################################
        # Q_factors_int = []
        # for U in Q_factors:
        #     Q_factors_int.append(U/self.scale_w)

        # self.Q_tensor = self.tensor.full_from_factors(Q_factors_int).reshape([self.out_features,self.in_features])
        # self.n_tensor = self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features])
        self.Q_factors = Q_factors
        self.output = output
        ### Code for test purpose only ##############################################
        
        return output
        
        # return F.linear(input,self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features]),Q_bias)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()



class Q_TensorizedLinear_module(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
                bit_w = 8,
                bit_b = 8,
                scale_w = 2**(-5),
                scale_b = 2**(-5),
                Q_back = False,
                recur = False,
    ):

        super(Q_TensorizedLinear_module,self).__init__()

        self.Q_back = Q_back
        self.recur = recur

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        self.bit_w = bit_w
        self.bit_b = bit_b

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        # self.max_q = 2.0**(bit-1)-1.0
        # self.min_q = -2.0**(bit-1)
        # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))

        if bias == False:
            self.bias = None
        else:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))


    def forward(self, input, rank_update=True):
        

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
        Q_factors = []        
        for U in self.tensor.factors:
            Q_factors.append(scale.apply(U,self.scale_w,self.bit_w, False))
        if self.Q_back:
            quant = lambda x : quant_func(x, bit=8)
            Q_factors = [torch.swapaxes(U,0,-1) for U in Q_factors[::-1]]
            output = ttm_times_mat.apply(quant,input.T, *Q_factors)
        elif self.recur:
            Q = lambda x: scale.apply(x,self.scale_w,self.bit_w, False)
            output = input @ self.tensor.full_from_factors(Q_factors,quant=Q).reshape([self.out_features,self.in_features]).T
        else:
            output = input @ self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features]).T

        if self.bias is not None:
            Q_bias = (scale.apply(self.bias,self.scale_b,self.bit_b, False))
            output = scale.apply(output,self.scale_b,self.bit_b, False) + Q_bias

        ### Code for test purpose only ##############################################
        # Q_factors_int = []
        # for U in Q_factors:
        #     Q_factors_int.append(U/self.scale_w)

        # self.Q_tensor = self.tensor.full_from_factors(Q_factors_int).reshape([self.out_features,self.in_features])
        # self.n_tensor = self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features])
        self.Q_factors = Q_factors
        self.output = output
        ### Code for test purpose only ##############################################
        
        return output
        
        # return F.linear(input,self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features]),Q_bias)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()
        
# class Q_conv2d_old(nn.Conv2d):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride= (1,1),
#                  padding=(0,0),
#                  dilation=(1,1),
#                  groups=1,
#                  bias = True,
#                  padding_mode = 'zeros',
#                  device=None,
#                  dtype=None,
#                  bit_w = 8,
#                  bit_b = 8,
#                  scale_w = 2**(-5),
#                  scale_b = 2**(-5)
#     ):
#         super(Q_conv2d_old,self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode,device,dtype)

#         self.stride = stride
#         self.padding = padding 
#         self.dilation = dilation
#         self.groups = groups

#         self.bit_w = bit_w
#         self.bit_b = bit_b
#         # self.max_q = 2.0**(bit-1)-1.0
#         # self.min_q = -2.0**(bit-1)
#         # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

#         self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
#         self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))
       

#     def forward(self, input):
#         Q_weight = scale.apply(self.weight,self.scale_w,self.bit_w)
#         Q_bias = scale.apply(self.bias,self.scale_b,self.bit_b)
        
#         output = F.conv2d(input,Q_weight,bias = None, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
#         output = scale.apply(output,self.scale_b,self.bit_b)
#         # print(output.shape)
#         # print(Q_bias.shape)
#         output = output.transpose(1,3)
#         output = output + Q_bias
#         output = output.transpose(1,3)

#         return output


class Q_conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride= (1,1),
                 padding=(0,0),
                 dilation=(1,1),
                 groups=1,
                 bias = True,
                 padding_mode = 'zeros',
                 device=None,
                 dtype=None,
                 bit_w = 8,
                 bit_b = 8,
                 scale_w = 2**(-5),
                 scale_b = 2**(-5)
    ):
        super(Q_conv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.transposed = transposed
        # self.output_padding = output_padding
        self.groups = groups

        self.bit_w = bit_w
        self.bit_b = bit_b
        # self.max_q = 2.0**(bit-1)-1.0
        # self.min_q = -2.0**(bit-1)
        # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init()
       
    def init(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        

    def forward(self, input):
        

        # output = F.conv2d(input,self.weight,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

        Q_weight = scale.apply(self.weight,self.scale_w,self.bit_w, False)
        
        output = F.conv2d(input,Q_weight,bias = None, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        

        self.output = output
        if self.bias is not None:
            Q_bias = scale.apply(self.bias,self.scale_b,self.bit_b, False)
            output = scale.apply(output,self.scale_b,self.bit_b,False)
            output = output.transpose(1,3)
            output = output + Q_bias
            output = output.transpose(1,3)

        self.Q_weight = Q_weight

        return output

class Q_Tensorizedconv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = (3,3),
                 stride= (1,1),
                 padding=(0,0),
                 dilation=(1,1),
                 groups=1,
                 bias = True,
                 padding_mode = 'zeros',
                 device=None,
                 dtype=None,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrain',
                 max_rank=20,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 eta = None,
                 bit_w = 8,
                 bit_b = 8,
                 scale_w = 2**(-5),
                 scale_b = 2**(-5)
    ):
        super(Q_Tensorizedconv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.transposed = transposed
        # self.output_padding = output_padding
        self.groups = groups

        self.bit_w = bit_w
        self.bit_b = bit_b
        # self.max_q = 2.0**(bit-1)-1.0
        # self.min_q = -2.0**(bit-1)
        # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))

        # self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init()

        if shape == None:
            shape = self.get_tensor_shape(out_channels)
            shape = shape + self.get_tensor_shape(in_channels)
            shape = shape + list(kernel_size)


        target_stddev = 2/np.sqrt(self.in_channels*kernel_size[0]*kernel_size[1])
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

       
    def init(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def get_tensor_shape(self,n):
        if n==64:
            return [8,8]
        if n==128:
            return [8,16]
        if n==256:
            return [16,16]
        if n==512:
            return [16,32]

    def forward(self, input, rank_update = True):
        
        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
        Q_factors = []        
        for U in self.tensor.factors:
            Q_factors.append(scale.apply(U,self.scale_w,self.bit_w, False))
        self.Q_factors = Q_factors
        
        # output = F.conv2d(input,self.weight,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

        w = self.tensor.get_full_factors(Q_factors).reshape(self.out_channels,self.in_channels,*self.kernel_size)
        output = F.conv2d(input,w,bias = None, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

        if self.bias is not None:
            Q_bias = (scale.apply(self.bias,self.scale_b,self.bit_b, False))
            output = scale.apply(output,self.scale_b,self.bit_b,False)

            self.output = output

            output = output.transpose(1,3)
            output = output + Q_bias
            output = output.transpose(1,3)

        self.Q_weight = w
        self.output = output

        return output


class Tensorizedconv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = (3,3),
                 stride= (1,1),
                 padding=(0,0),
                 dilation=(1,1),
                 groups=1,
                 bias = True,
                 padding_mode = 'zeros',
                 device=None,
                 dtype=None,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrain',
                 max_rank=20,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 eta = None,
    ):
        super(Tensorizedconv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.transposed = transposed
        # self.output_padding = output_padding
        self.groups = groups

    

        # self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init()

        if shape == None:
            shape = self.get_tensor_shape(out_channels)
            shape = shape + self.get_tensor_shape(in_channels)
            shape = shape + list(kernel_size)


        target_stddev = 2/np.sqrt(self.in_channels*kernel_size[0]*kernel_size[1])
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

       
    def init(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def get_tensor_shape(self,n):
        if n==64:
            return [8,8]
        if n==128:
            return [8,16]
        if n==256:
            return [16,16]
        if n==512:
            return [16,32]

    def forward(self, input, rank_update = True):
        
        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
       
        # output = F.conv2d(input,self.weight,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

        w = self.tensor.get_full().reshape(self.out_channels,self.in_channels,*self.kernel_size)
        output = F.conv2d(input,w,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        self.output = output
        return output
