from collections import OrderedDict
import torch
import torch.nn as nn

from core.fsr_mrr.fsr_mrr_linear import FSR_MRRLinear

__all__ = [
    "MRR_FCBlock",
]

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

def xavier_init(layer):
    with torch.no_grad():
        if type(layer) == nn.Linear:
            if hasattr(layer, 'weight'):
                nn.init.xavier_normal_(layer.weight)
        else:
            raise TypeError(f'Expecting nn.Linear got type={type(layer)} instead')

nl_init_dict = dict(
    sine=(Sine(), xavier_init),
    silu=(nn.SiLU(), xavier_init),
    tanh=(nn.Tanh(), xavier_init),
    relu=(nn.ReLU(), xavier_init),
    gelu=(nn.GELU(), xavier_init)
)


class MRR_FCBlock(nn.Module):

    def __init__(
        self, 
        in_features=3, 
        out_features=1, 
        hidden_features=20, 
        num_layers=3, 
        nonlinearity='sine',
        bias=False, 
        in_bit=32,
        mode='weight',
        mrr_config_type='single-posneg', # 'single-posneg', 'single-pos', 'multi-posneg', 'multi-pos'
        device=None, 
        ) -> None:
        super(MRR_FCBlock, self).__init__()

        nl, init = nl_init_dict[nonlinearity]
        
        if mode == 'laser':
            from .fsr_mrr_array_laser import FSR_MRR_Config_Laser
            self.MRRConfig = FSR_MRR_Config_Laser()
        elif mode == 'voltage':
            from .fsr_mrr_array_simu import FSR_MRR_Config_Simu
            self.MRRConfig = FSR_MRR_Config_Simu(mrr_config_type, device=device)
        else:
            self.MRRConfig = None

        # ====================== BUILD LAYERS ======================
        self.net = OrderedDict()
        
        # input layer
        self.net['fc1'] = FSR_MRRLinear(
            in_features = in_features,
            out_features = hidden_features,
            bias = bias,
            in_bit = in_bit,
            mode = mode,
            MRRConfig = self.MRRConfig,
            device = device,
        )

        self.net['nl1'] = nl

        # hidden layers
        for i in range(2, num_layers):
            self.net['fc%d' % i] = FSR_MRRLinear(
                in_features = hidden_features,
                out_features = hidden_features,
                bias = bias,
                in_bit = in_bit,
                mode = mode,
                MRRConfig = self.MRRConfig,
                device = device,
            )
        
            self.net['nl%d' % i] = nl

        # output layer
        # self.net['fc%d' % num_layers] = FSR_MRRLinear(
        #     in_features = hidden_features,
        #     out_features = out_features,
        #     bias = bias,
        #     # in_bit = in_bit,
        #     in_bit = 32,
        #     mode = mode,
        #     MRRConfig = self.MRRConfig,  
        #     device = device,
        # )
        self.net['fc%d' % num_layers] = nn.Linear(in_features=hidden_features, out_features=out_features, bias=bias, device=device)
        
        self.net = nn.Sequential(self.net)

    def forward(self, x):
        return self.net(x)