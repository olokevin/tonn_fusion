from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.types import Device
__all__ = [
    "FSR_MRRLinear",
    "FSR_MRRBlockLinear",
]



class FSR_MRRLinear(nn.Module):
    """
    Linear layer constructed by cascaded AddDropMRRs.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    __annotations__ = {"bias": Optional[Tensor]}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        in_bit: int = 32,
        mode: str = "weight",
        MRRConfig=None,
        device: Device = torch.device("cpu"),
    ):
        super(FSR_MRRLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.device = device
        self.MRRConfig = MRRConfig
        
        self.v_min = self.MRRConfig.v_min
        self.v_max = self.MRRConfig.v_max
        self.fsr_channels = self.MRRConfig.fsr_channels
        self.mrr_in = self.MRRConfig.mrr_in
        self.mrr_out = self.MRRConfig.mrr_out
        self.pos_and_neg = self.MRRConfig.pos_and_neg
        
        self.grid_dim_x = int(np.ceil(self.in_features / self.mrr_in))
        self.grid_dim_y = int(np.ceil(self.out_features / self.mrr_out))
        self.in_features_pad = self.grid_dim_x * self.mrr_in
        self.out_features_pad = self.grid_dim_y * self.mrr_out
        
        self.x_zero_pad = None
        
        # self.v_max = 10.8
        # self.v_pi = 4.36
        # self.gamma = np.pi / self.v_pi**2
        # self.w_bit = 32
        self.in_bit = in_bit

        ### build trainable parameters
        self.build_parameters(mode)
        ### quantization tool. if self.in_bit is 
        self.input_quantizer = quantize_fn(bits=self.in_bit)
        # self.weight_quantizer = weight_quantize_fn(self.w_bit, alg="dorefa_sym")
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        
        ### weight saved in build_parameters ###

    def build_parameters(self, mode: str = "weight") -> None:
        # TIA gain
        S_scale = torch.ones(1, device=self.device, dtype=torch.float32)

        if mode == "weight":
            weight = torch.empty(
                self.out_features_pad,
                self.in_features_pad,
                device=self.device,
            )
            self.weight = Parameter(weight, requires_grad=True)
        elif mode == "phase":
            raise NotImplementedError
        elif mode == "voltage":
            voltage = torch.empty(
                self.out_features_pad,
                self.in_features_pad,
                device=self.device,
            )
            self.voltage = Parameter(voltage, requires_grad=False)
            # self.S_scale = Parameter(S_scale)
            self.register_buffer('S_scale', S_scale)
            self.weight = self.build_weight()
            # self.register_buffer('weight', self.build_weight())
        elif mode == "laser":
            voltage = torch.empty(
                self.out_features,
                self.in_features,
                device=self.device,
            )
            self.voltage = Parameter(voltage, requires_grad=False)
        else:
            raise NotImplementedError

        # for p_name, p in {
        #     "weight": weight,
        #     "voltage": voltage,
        #     "S_scale": S_scale,
        # }.items():
        #     if not hasattr(self, p_name):
        #         self.register_buffer(p_name, p)

    def reset_parameters(self) -> None:
        if self.mode in {"weight"}:
            init.kaiming_normal_(self.weight.data)
        elif self.mode in {"voltage", "laser"}:
            init.uniform_(self.voltage.data, self.MRRConfig.v_min, self.MRRConfig.v_max)
            # scale = self.weight.data.abs().max()
            # self.S_scale.data.fill_(scale)
            
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)
    
    def build_weight(self) -> Tensor:
        if self.mode == "weight":
            weight = self.weight
        elif self.mode == "phase":
            raise NotImplementedError
        elif self.mode == "voltage":
            weight = self.build_weight_from_voltage(self.voltage, self.S_scale)
        elif self.mode == "laser":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return weight
    
    def commit_all(self) -> None:
        if self.mode == "voltage":
            self.voltage.clamp(self.v_min, self.v_max)
            self.weight = self.build_weight_from_voltage(self.voltage, self.S_scale)
        elif self.mode == "laser":
            self.voltage.clamp(self.v_min, self.v_max)
        else:
            raise NotImplementedError
    
    def commit_coordinate(self, idx):
        if self.mode == "voltage":
            self.voltage.clamp(self.v_min, self.v_max)
            
            block_out_idx = (idx // self.in_features_pad) // self.mrr_out
            block_in_idx = (idx % self.in_features_pad) // self.mrr_in
            
            # the voltage has been updated outside
            voltage = self.voltage[block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in]
            
            ##### Multi FSR
            if self.MRRConfig.fsr_channels > 1:
                if self.pos_and_neg:
                    ### 0813 setting
                    pos_voltage = voltage * (voltage >= 0).float()
                    neg_voltage = (voltage * (voltage < 0).float()).abs()
                    
                    pos_weight = self.MRRConfig.Four_MRR_array(pos_voltage)
                    neg_weight = self.MRRConfig.Four_MRR_array(neg_voltage)
                    
                    self.weight[0][:, block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in].data.copy_(pos_weight)
                    self.weight[1][:, block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in].data.copy_(neg_weight)
                    
                    ### 0812 setting
                    # MMR_weight = self.MRRConfig.Four_MRR_array(voltage.abs())
                    
                    # pos_weight = MMR_weight * torch.stack([voltage > 0] * self.MRRConfig.fsr_channels, dim=0)
                    # neg_weight = MMR_weight * torch.stack([voltage <= 0] * self.MRRConfig.fsr_channels, dim=0)
                    
                    # self.weight[0][:, block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in].data.copy_(pos_weight)
                    # self.weight[1][:, block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in].data.copy_(neg_weight)
                  
                else:
                    voltage = voltage * (voltage >= 0).float()
                    MMR_weight = self.MRRConfig.Four_MRR_array(voltage)
                    self.weight[:, block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in].data.copy_(MMR_weight)
            ##### Single FSR
            else:
                if self.pos_and_neg:
                    ### 0813 setting
                    pos_voltage = voltage * (voltage >= 0).float()
                    neg_voltage = (voltage * (voltage < 0).float()).abs()
                    
                    pos_weight = self.MRRConfig.One_MRR_array(pos_voltage)
                    neg_weight = self.MRRConfig.One_MRR_array(neg_voltage)
                    
                    self.weight[0][block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in].data.copy_(pos_weight)
                    self.weight[1][block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in].data.copy_(neg_weight)
                    
                    ### 0812 setting
                    # MMR_weight = self.MRRConfig.One_MRR_array(voltage.abs())
                    
                    # pos_weight = MMR_weight * (voltage > 0).float()
                    # neg_weight = MMR_weight * (voltage <= 0).float()
                    
                    # self.weight[0][block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in].data.copy_(pos_weight)
                    # self.weight[1][block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in].data.copy_(neg_weight)
                  
                else:
                    voltage = voltage * (voltage >= 0).float()
                    MMR_weight = self.MRRConfig.One_MRR_array(voltage)
                    self.weight[block_out_idx*self.mrr_out : (block_out_idx+1)*self.mrr_out, block_in_idx*self.mrr_in : (block_in_idx+1)*self.mrr_in].data.copy_(MMR_weight)
        elif self.mode == "laser":
            self.voltage.clamp(self.v_min, self.v_max)
        else:
            raise NotImplementedError
          
    def build_weight_from_voltage(
        self,
        voltage: Tensor,
        S_scale: Tensor,
    ) -> Tensor:
        ### Multi FSR
        if self.MRRConfig.fsr_channels > 1:
            if self.pos_and_neg:
                pos_weight = torch.empty(
                    self.MRRConfig.fsr_channels,
                    self.out_features_pad,
                    self.in_features_pad,
                    device=self.device,
                )
                neg_weight = torch.empty(
                    self.MRRConfig.fsr_channels,
                    self.out_features_pad,
                    self.in_features_pad,
                    device=self.device,
                )
               
                ### 0813 setting
                pos_voltage = voltage * (voltage >= 0).float()
                neg_voltage = (voltage * (voltage < 0).float()).abs()
                                
                for i in range(self.grid_dim_y):
                    for j in range(self.grid_dim_x):
                        pos_v = pos_voltage[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in]
                        pos_weight[:, i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in] = self.MRRConfig.Four_MRR_array(pos_v)
                        neg_v = neg_voltage[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in]
                        neg_weight[:, i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in] = self.MRRConfig.Four_MRR_array(neg_v)
                
                ### 0812 setting
                # pos_voltage = voltage.abs()
                
                # for i in range(self.grid_dim_y):
                #     for j in range(self.grid_dim_x):
                #         v = pos_voltage[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in]
                #         pos_weight[:, i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in] = self.MRRConfig.Four_MRR_array(v)

                # neg_weight = pos_weight * torch.stack([voltage <= 0] * self.MRRConfig.fsr_channels, dim=0)
                # pos_weight = pos_weight * torch.stack([voltage > 0] * self.MRRConfig.fsr_channels, dim=0)
                
                return pos_weight, neg_weight
            else:
                weight = torch.empty(
                    self.MRRConfig.fsr_channels,
                    self.out_features_pad,
                    self.in_features_pad,
                    device=self.device,
                )
                voltage = voltage * (voltage >= 0).float()
                for i in range(self.grid_dim_y):
                    for j in range(self.grid_dim_x):
                        v = voltage[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in]
                        weight[:, i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in] = self.MRRConfig.Four_MRR_array(v)

                return weight
        ### Single FSR
        else:
            if self.pos_and_neg:
                pos_weight = torch.empty(
                    self.out_features_pad,
                    self.in_features_pad,
                    device=self.device,
                )
                neg_weight = torch.empty(
                    self.out_features_pad,
                    self.in_features_pad,
                    device=self.device,
                )
               
                ### 0813 setting
                pos_voltage = (voltage * (voltage >= 0)).abs()
                neg_voltage = (voltage * (voltage < 0)).abs()
                
                for i in range(self.grid_dim_y):
                    for j in range(self.grid_dim_x):
                        pos_v = pos_voltage[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in]
                        pos_weight[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in] = self.MRRConfig.One_MRR_array(pos_v)
                        
                        neg_v = neg_voltage[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in]
                        neg_weight[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in] = self.MRRConfig.One_MRR_array(neg_v)
                
                ### 0812 setting
                # pos_voltage = voltage.abs()
                
                # for i in range(self.grid_dim_y):
                #     for j in range(self.grid_dim_x):
                #         v = pos_voltage[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in]
                #         pos_weight[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in] = self.MRRConfig.One_MRR_array(v)

                # neg_weight = pos_weight * (voltage <= 0).float()
                # pos_weight = pos_weight * (voltage > 0).float()
                
                return pos_weight, neg_weight
            else:
                weight = torch.empty(
                    self.out_features_pad,
                    self.in_features_pad,
                    device=self.device,
                )
                voltage = (voltage * (voltage >= 0)).abs()
                for i in range(self.grid_dim_y):
                    for j in range(self.grid_dim_x):
                        v = voltage[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in]
                        weight[i*self.mrr_out : (i+1)*self.mrr_out, j*self.mrr_in : (j+1)*self.mrr_in] = self.MRRConfig.One_MRR_array(v)

                return weight

    def build_voltage_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        return NotImplementedError

    # def sync_parameters(self, src: str = "weight") -> None:
    #     """
    #     description: synchronize all parameters from the source parameters
    #     """
    #     if src == "weight":
    #         self.build_voltage_from_weight(self.weight)
    #     elif src == "phase":
    #         raise NotImplementedError
    #     elif src == "voltage":
    #         # TODO quantizer
    #         # TODO add noise
    #         self.weight= self.build_weight_from_voltage(
    #             self.voltage,
    #             self.S_scale,
    #         )
    #     else:
    #         raise NotImplementedError

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_quantizer.set_bitwidth(w_bit)
        self.weight_quantizer.set_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bitwidth(in_bit)

    # def load_parameters(self, param_dict: Dict[str, Any]) -> None:
    #     """
    #     description: update parameters based on this parameter dictionary\\
    #     param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
    #     """
    #     super().load_parameters(param_dict=param_dict)
    #     if self.mode == "phase":
    #         self.build_weight(update_list=param_dict)

    # @classmethod
    # def from_layer(
    #     cls,
    #     layer: nn.Linear,
    #     mode: str = "weight",
    #     MRRConfig=MRRConfig_5um_HQ,
    # ) -> nn.Module:
    #     """Initialize from a nn.Linear layer. Weight mapping will be performed

    #     Args:
    #         mode (str, optional): parametrization mode. Defaults to "weight".
    #         decompose_alg (str, optional): decomposition algorithm. Defaults to "clements".
    #         photodetect (bool, optional): whether to use photodetect. Defaults to True.

    #     Returns:
    #         Module: a converted AddDropMRRLinear module
    #     """
    #     assert isinstance(
    #         layer, nn.Linear
    #     ), f"The conversion target must be nn.Linear, but got {type(layer)}."
    #     in_features = layer.in_features
    #     out_features = layer.out_features
    #     bias = layer.bias is not None
    #     device = layer.weight.data.device
    #     instance = cls(
    #         in_features=in_features,
    #         out_features=out_features,
    #         bias=bias,
    #         mode=mode,
    #         MRRConfig=MRRConfig,
    #         device=device,
    #     ).to(device)
    #     instance.weight.data.copy_(layer.weight)
    #     instance.sync_parameters(src="weight")
    #     if bias:
    #         instance.bias.data.copy_(layer.bias)

    #     return instance
      
    def forward(self, x: Tensor) -> Tensor:
        # if self.in_bit <= 16:
        x = self.input_quantizer(x)
        
        if self.in_features_pad > self.in_features:
            if self.x_zero_pad is None or self.x_zero_pad.size(0) != x.size(0):
                # self.x_zero_pad = torch.zeros(
                #     x.size(0), self.in_channel_pad - self.in_channel, device=x.device, dtype=x.dtype
                # )
                self.x_zero_pad = torch.zeros(
                    (x.shape[:-1])+((self.in_features_pad - self.in_features),), device=x.device, dtype=x.dtype
                )
            x = torch.cat([x, self.x_zero_pad], dim=-1)
            
        # if not self.fast_forward_flag or self.weight is None:
        #     weight = self.build_weight()  # [out_features, in_features]
        # else:
        #     weight = self.weight

        # if self.bias is not None:
        #     if self.out_features_pad > self.out_features:
        #         bias = torch.cat([self.bias, torch.zeros(self.out_features_pad - self.out_features, device=self.bias.device)])
        #     else:
        #         bias = self.bias
        # else:
        #     bias = None

        if self.mode == "weight":
            if self.out_features_pad > self.out_features:
                bias = torch.cat([self.bias, torch.zeros(self.out_features_pad - self.out_features, device=self.bias.device)])
            else:
                bias = self.bias
            out = F.linear(
                    x,
                    self.weight,
                    bias=bias,
                )
            out = out[..., : self.out_features]
            # if self.in_bit <= 16:
            out = self.input_quantizer(out)
        elif self.mode == "voltage":
            ##### Multi FSR
            if self.fsr_channels > 1:
                # weight: [fsr_channels, out_features_pad, in_features_pad]
                # x: [bz, ..., in_features_pad]
                x_shape = x.shape
                x = x.reshape(-1, self.in_features_pad)
                batch_sz = x.shape[0]
                fsr_groups = int(np.ceil(batch_sz / self.fsr_channels))
                batch_sz_pad = self.fsr_channels * fsr_groups
                if batch_sz_pad > batch_sz:
                    x = torch.cat([x, torch.zeros(batch_sz_pad - batch_sz, self.in_features_pad, device=x.device)], dim=0)
                
                x = x.reshape(self.fsr_channels, fsr_groups, self.in_features_pad)
                out = torch.empty(self.fsr_channels, fsr_groups, self.out_features_pad, device=x.device)
                if self.pos_and_neg:
                    pos_weight, neg_weight = self.weight
                    for fsr_channel in range(self.fsr_channels):
                        out[fsr_channel] = F.linear(
                            x[fsr_channel],
                            pos_weight[fsr_channel],
                            bias=None,
                            # bias=bias,
                        ) - F.linear(
                            x[fsr_channel],
                            neg_weight[fsr_channel],
                            bias=None,
                        )
                
                else:
                    for fsr_channel in range(self.fsr_channels):
                        out[fsr_channel] = F.linear(
                            x[fsr_channel],
                            weight[fsr_channel],
                            bias=None,
                            # bias=bias,
                        )
                
                out = out.reshape(-1, self.out_features_pad).reshape(x_shape[:-1]+(self.out_features_pad,))
            ##### Single FSR
            elif self.fsr_channels == 1:
                if self.pos_and_neg:
                    pos_weight, neg_weight = self.weight
                    out = F.linear(
                        x,
                        pos_weight,
                        bias=None,
                        # bias=bias,
                    ) - F.linear(
                        x,
                        neg_weight,
                        bias=None,
                    )
                else:
                    weight = self.weight
                    out = F.linear(
                        x,
                        weight,
                        bias=None,
                        # bias=bias,
                    )
        
            out = out[..., : self.out_features]
            # if self.in_bit <= 16:
            out = self.input_quantizer(out)
            out = out + self.bias
        
        elif self.mode == "laser":
            if self.in_features_pad > self.in_features:
                voltage = torch.cat([self.voltage, torch.zeros((self.out_features, self.in_features_pad - self.in_features), device=self.voltage.device)], dim=1)
            if self.out_features_pad > self.out_features:
                voltage = torch.cat([self.voltage, torch.zeros((self.out_features_pad - self.out_features, self.in_features_pad), device=self.voltage.device)], dim=0)
            
            out = self.MRRConfig.linear(x,voltage)
            out = out[..., : self.out_features]
            out = self.input_quantizer(out)
            out = out + self.bias
        return out

def get_weight_scales(w, n_bit=8, k_near_zero_tolerance=1e-6, allow_all_same=False):
    # NOTICE: the zero point for w is always chosen as 0, so it is actually a symmetric quantization
    def _extract_min_max_from_weight(weights):
        dim_size = weights.shape[0]

        if weights.max() == weights.min():  # all the elements are the same?
            mins = np.zeros(dim_size)
            maxs = np.zeros(dim_size)
            single_value = weights.min().item()
            if single_value < 0.:
                mins[:] = single_value
                maxs[:] = -single_value
            elif single_value > 0.:
                mins[:] = -single_value
                maxs[:] = single_value
            else:
                mins[:] = maxs[:] = single_value
            return torch.from_numpy(mins).to(weights.device), torch.from_numpy(maxs).to(weights.device)
        else:
            weights = weights.reshape(weights.shape[0], -1)
            mins = weights.min(dim=1)[0]
            maxs = weights.max(dim=1)[0]
            maxs = torch.max(mins.abs(), maxs.abs())
            mins = -maxs
            return mins, maxs

    def _expand_very_small_range(mins, maxs):
        k_smallest_half_range = k_near_zero_tolerance / 2
        if (maxs - mins).min() > k_near_zero_tolerance:
            return mins, maxs
        else:
            for i in range(len(mins)):
                mins[i] = min(mins[i], -k_smallest_half_range)
                maxs[i] = max(maxs[i], k_smallest_half_range)
            return mins, maxs

    mins, maxs = _extract_min_max_from_weight(w)
    mins, maxs = _expand_very_small_range(mins, maxs)
    assert (mins + maxs).max() < 1e-9  # symmetric
    return maxs / (2 ** (n_bit - 1) - 1)

class quantize_fn(object):
    def __init__(self, bits):
        super(quantize_fn).__init__()
        self.bits = bits

    def __call__(self, input):
        if self.bits == 32:
            out = input
        elif self.bits == 1:
            out = np.sign(input)
        ### dynamic quantization
        elif self.bits > 1 and self.bits <= 16:
            scales = get_weight_scales(out, n_bit=self.bits)
            out = (out / scales.view(-1, 1, 1, 1)).round() * scales.view(-1, 1, 1, 1)
        ### decimal bits
        elif self.bits >= 100 and (self.bits % 10 == 0):
            out = torch.round(input * self.bits) / self.bits
        else:
            raise NotImplementedError('Unsupported bitwidth')
        return out