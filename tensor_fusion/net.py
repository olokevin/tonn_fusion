
from tensor_fusion.subnet import *
from tensor_fusion.module import TensorFusion, AdaptiveRankFusion, LowRankFusion
from tensor_fusion.util import *

class TFN(nn.Module):

    def __init__(self, input_sizes, hidden_sizes, fusion_size, out_size, dropouts, device=None, dtype=None):

        super().__init__()

        self.text_subnet = TextSubNet(input_sizes[0], hidden_sizes[0], hidden_sizes[0], dropout=dropouts[0], 
                                      device=device, dtype=dtype)

        self.audio_subnet = SubNet(input_sizes[1], hidden_sizes[1], dropouts[1], device=device, dtype=dtype)

        self.video_subnet = SubNet(input_sizes[2], hidden_sizes[2], dropouts[2], device=device, dtype=dtype)

        fusion_input_sizes = tuple([x+1 for x in hidden_sizes])
        self.fusion_layer = TensorFusion(fusion_input_sizes, fusion_size, dropouts[3], device=device, dtype=dtype)

        self.inference_subnet = InferenceSubNet(fusion_size, out_size, dropouts[4], device=device, dtype=dtype)

    def forward(self, text_in, audio_in, video_in):

        text_out = self.text_subnet(text_in)
        audio_out = self.audio_subnet(audio_in)
        video_out = self.video_subnet(video_in)

        fusion_inputs = concatenate_one([text_out, audio_out, video_out])

        output = self.fusion_layer(fusion_inputs)

        output = self.inference_subnet(output)

        return output
    
    def count_parameters(self):

        param_list = list(self.parameters())

        count = 0
        for param in param_list:
            count += torch.numel(param)
    
        return count

    def count_fusion_parameters(self):

        count = 0
        for param in list(self.fusion_layer.parameters()):
            count += torch.numel(param)

        return count

class LMF(nn.Module):

    def __init__(self, input_sizes, hidden_sizes, fusion_size, rank, out_size, dropouts, device=None, dtype=None):

        super().__init__()

        self.text_subnet = TextSubNet(input_sizes[0], hidden_sizes[0], hidden_sizes[0], dropout=dropouts[0], 
                                      device=device, dtype=dtype)

        self.audio_subnet = SubNet(input_sizes[1], hidden_sizes[1], dropouts[1], device=device, dtype=dtype)

        self.video_subnet = SubNet(input_sizes[2], hidden_sizes[2], dropouts[2], device=device, dtype=dtype)

        fusion_input_sizes = tuple([x+1 for x in hidden_sizes])
        self.fusion_layer = LowRankFusion(fusion_input_sizes, fusion_size, rank, dropouts[3], device=device, dtype=dtype)

        self.inference_subnet = InferenceSubNet(fusion_size, out_size, dropouts[4], device=device, dtype=dtype)
    
    def forward(self, text_in, audio_in, video_in):

        text_out = self.text_subnet(text_in)
        audio_out = self.audio_subnet(audio_in)
        video_out = self.video_subnet(video_in)

        fusion_inputs = concatenate_one([text_out, audio_out, video_out])

        output = self.fusion_layer(fusion_inputs)

        output = self.inference_subnet(output)

        return output
        
    def count_parameters(self):

        param_list = list(self.parameters())

        count = 0
        for param in param_list:
            count += torch.numel(param)
    
        return count

    def count_fusion_parameters(self):

        count = 0
        for param in list(self.fusion_layer.parameters()):
            count += torch.numel(param)

        return count

class ARF(nn.Module):

    def __init__(self, input_sizes, hidden_sizes, fusion_size, max_rank, out_size, dropouts, 
                 prior_type='log_uniform', eta=None, device=None, dtype=None):

        super().__init__()

        self.fusion_size = fusion_size
        self.prior_type = prior_type
        self.eta = eta
        self.dtype = dtype
        self.device = device

        self.text_subnet = TextSubNet(input_sizes[0], hidden_sizes[0], hidden_sizes[0], dropout=dropouts[0], 
                                      device=device, dtype=dtype)

        self.audio_subnet = SubNet(input_sizes[1], hidden_sizes[1], dropouts[1], device=device, dtype=dtype)

        self.video_subnet = SubNet(input_sizes[2], hidden_sizes[2], dropouts[2], device=device, dtype=dtype)

        self.fusion_input_sizes = tuple([x+1 for x in hidden_sizes])
        
        self.fusion_layer = AdaptiveRankFusion(self.fusion_input_sizes, fusion_size, dropouts[3],
                                               max_rank=max_rank, prior_type=prior_type, eta=eta,
                                               device=device, dtype=dtype)

        self.inference_subnet = InferenceSubNet(fusion_size, out_size, dropouts[4], device=device, dtype=dtype)
    
    def forward(self, text_in, audio_in, video_in):

        text_out = self.text_subnet(text_in)
        audio_out = self.audio_subnet(audio_in)
        video_out = self.video_subnet(video_in)

        fusion_inputs = concatenate_one([text_out, audio_out, video_out])

        output = self.fusion_layer(fusion_inputs)

        output = self.inference_subnet(output)

        return output

    def get_log_prior(self):

        return self.fusion_layer.get_log_prior()

    def compress(self):

        columns = []
        for factor in self.fusion_layer.weight_tensor_factors:
            columns_ = torch.where(factor.var(axis=0) > 1e-5)[0].tolist()
            [columns.append(x) for x in columns_ if x not in columns]
        
        new_factors = []
        for factor in self.fusion_layer.weight_tensor_factors:
            new_factors.append(factor[:,columns].data)
        
        rank = len(columns)
        new_fusion_layer = AdaptiveRankFusion(self.fusion_input_sizes, self.fusion_size,
                                               max_rank=rank, prior_type=self.prior_type, eta=self.eta,
                                               device=self.device, dtype=self.dtype)

        for i in range(4):
            new_fusion_layer.weight_tensor_factors[i].data = new_factors[i]

        self.fusion_layer = new_fusion_layer

    def count_parameters(self):

        param_list = list(self.parameters())

        count = 0
        for param in param_list:
            count += torch.numel(param)
    
        return count

    def count_fusion_parameters(self):

        count = 0
        for param in list(self.fusion_layer.parameters()):
            count += torch.numel(param)

        return count

class ARF_with_AR_TextSubNet(nn.Module):

    def __init__(self, input_sizes, hidden_sizes, fusion_size, max_ranks, out_size, dropouts, 
                 tensor_type='TT', prior_type='log_uniform', eta=None, device=None, dtype=None):

        super().__init__()

        self.fusion_size = fusion_size
        self.prior_type = prior_type
        self.eta = eta
        self.dtype = dtype
        self.device = device

        self.text_subnet = AdaptiveRankTextSubNet(input_sizes[0], hidden_sizes[0], hidden_sizes[0], dropout=dropouts[0], 
                                                  max_rank=max_ranks[0], tensor_type=tensor_type, prior_type=prior_type, eta=eta,
                                                  device=device, dtype=dtype)

        self.audio_subnet = SubNet(input_sizes[1], hidden_sizes[1], dropouts[1], device=device, dtype=dtype)

        self.video_subnet = SubNet(input_sizes[2], hidden_sizes[2], dropouts[2], device=device, dtype=dtype)

        self.fusion_input_sizes = tuple([x+1 for x in hidden_sizes])
        
        self.fusion_layer = AdaptiveRankFusion(self.fusion_input_sizes, fusion_size, dropouts[3],
                                               max_rank=max_ranks[1], prior_type='log_uniform', eta=None,
                                               device=device, dtype=dtype)

        self.inference_subnet = InferenceSubNet(fusion_size, out_size, dropouts[4], device=device, dtype=dtype)
    
    def forward(self, text_in, audio_in, video_in):

        text_out = self.text_subnet(text_in)
        audio_out = self.audio_subnet(audio_in)
        video_out = self.video_subnet(video_in)

        fusion_inputs = concatenate_one([text_out, audio_out, video_out])

        output = self.fusion_layer(fusion_inputs)

        output = self.inference_subnet(output)

        return output

    def get_log_prior(self):

        return self.fusion_layer.get_log_prior() + self.text_subnet.get_log_prior()

    def count_parameters(self):

        param_list = list(self.parameters())

        count = 0
        for param in param_list:
            count += torch.numel(param)
    
        return count

    def count_fusion_parameters(self):

        count = 0
        for param in list(self.fusion_layer.parameters()):
            count += torch.numel(param)

        return count