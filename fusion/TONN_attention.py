import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyutils.config import configs
from core.models.sparse_bp_ttm_mlp import TTM_Linear_module

class TONN_Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=True,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True,
            uncompressed=0,
            embedded=True):

        super().__init__()

        self.embedded = embedded
        
        if self.embedded == False:
          raise ValueError("Should have been embedded")
          # tensor_shape = [[16,16,8,5],[4,4,8,4]]
          # if tensorized:
          #     self.src_word_emb = TensorizedEmbedding(
          #             tensor_type=emb_tensor_type,
          #             max_rank=emb_rank,
          #             shape=emb_shape,
          #             prior_type='log_uniform',
          #             eta=None)
          
          # else:
          #     self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)

        # self.src_word_emb = model_origin._modules['distilbert']._modules['embeddings']
      
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        # self.position_enc = nn.Embedding(n_position,d_word_vec)
        self.token_type_emb = nn.Embedding(2,d_word_vec)

        self.register_buffer("position_ids", torch.arange(n_position).expand((1, -1)))

        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            TONN_EncoderLayer(d_model, d_inner, n_head,d_q, d_k, d_v, dropout=dropout,
            attention_shape=attention_shape,attention_rank=attention_rank,attention_tensor_type=attention_tensor_type,
            ffn_shape=ffn_shape,ffn_rank=ffn_rank,ffn_tensor_type=ffn_tensor_type,
            tensorized=tensorized,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, seg=None, ranks=None, scale=None):

        enc_slf_attn_list = []

        # -- Forward
        if self.embedded == False:
          enc_output = self.src_word_emb(src_seq)
        else:
          enc_output = src_seq

        # print("src_seq dimension is: {}".format(src_seq.shape))
        
        # position_ids = self.position_ids[:, 0: src_seq.shape[1]]

        # print("position_ids dimension is: {}".format(position_ids.shape))

        # if seg==None:
        #     seg = torch.zeros(src_mask.shape).to(int).to(src_mask.device)
        
        # print("seg dimension is: {}".format(seg.shape))
        # print("position_enc dimension is: {}".format(self.position_enc.shape))
        # print("token_type_emb dimension is: {}".format(self.token_type_emb.shape))

        # print(position_ids)
        

        # enc_output = enc_output+self.position_enc(position_ids) +self.token_type_emb(seg)
        enc_output = enc_output+self.position_enc(enc_output)

        enc_output = self.layer_norm(enc_output)
        enc_output = self.dropout(enc_output)
        

        for i,enc_layer in enumerate(self.layer_stack):
            if ranks!=None:
                # print(ranks[i])
                enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask, ranks=ranks[i],scale=scale)
            else:
                enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)

            # enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        return enc_output

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # print(self.pos_table[:, :x.size(1),:2])
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TONN_EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_q,d_k, d_v, dropout=0.1,
                attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
                ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
                bit_attn = 8, scale_attn = 2**(-5), 
                bit_ffn = 8, scale_ffn = 2**(-5),
                bit_a = 8, scale_a = 2**(-5),
                quantized = False,
                tensorized=True):
        super(TONN_EncoderLayer, self).__init__()
        
        if tensorized:
            self.slf_attn = Tensor_Attention(d_model,d_q,d_k,d_v,n_head, dropout=dropout,shape=attention_shape,rank=attention_rank,tensor_type=attention_tensor_type,
            bit_w=bit_attn,scale_w=scale_attn,
            quantized=quantized)
            self.pos_ffn = Tensor_PFF(d_model, d_inner, dropout=dropout,shape=ffn_shape,rank=ffn_rank,tensor_type=ffn_tensor_type,
            bit_w=bit_ffn,scale_w=scale_ffn,
            bit_a=bit_a, scale_a=scale_a,
            quantized=quantized)
        else:
            raise NotImplementedError
            # self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            # self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
        
       
    def forward(self, enc_input, slf_attn_mask=None, ranks=None,scale=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, ranks=ranks, scale=scale)
        enc_output = self.pos_ffn(enc_output,ranks=ranks,scale=scale)
        return enc_output, enc_slf_attn

class Tensor_Attention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model,d_q,d_k,d_v, n_head, shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], rank=[20,20],tensor_type = 'TensorTrain', dropout=0.1,
                bit_w = 8, scale_w = 2**(-5), 
                quantized = False):
        super().__init__()


        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

      
        em_stepsize = 1.0

        assert quantized == False
        
        self.w_qs = TTM_Linear_module(
                in_features=d_model, 
                out_features=d_q*n_head, 
                bias=True, 
                shape=shape[0], 
                tensor_type='TensorTrainMatrix', 
                max_rank=rank[0],
                em_stepsize=em_stepsize,
                miniblock=configs.model.miniblock,
                mode=configs.model.mode,
                v_max=configs.quantize.v_max,
                v_pi=configs.quantize.v_pi,
                in_bit=configs.quantize.in_bit,
                w_bit=configs.quantize.w_bit,
                photodetect=False,
                activation=False,
                act_thres=configs.model.act_thres
            )
        self.w_ks = TTM_Linear_module(
                in_features=d_model, 
                out_features=d_k*n_head, 
                bias=True, 
                shape=shape[0], 
                tensor_type='TensorTrainMatrix', 
                max_rank=rank[0],
                em_stepsize=em_stepsize,
                miniblock=configs.model.miniblock,
                mode=configs.model.mode,
                v_max=configs.quantize.v_max,
                v_pi=configs.quantize.v_pi,
                in_bit=configs.quantize.in_bit,
                w_bit=configs.quantize.w_bit,
                photodetect=False,
                activation=False,
                act_thres=configs.model.act_thres
            )
        self.w_vs = TTM_Linear_module(
                in_features=d_model, 
                out_features=d_v*n_head, 
                bias=True, 
                shape=shape[0], 
                tensor_type='TensorTrainMatrix', 
                max_rank=rank[0],
                em_stepsize=em_stepsize,
                miniblock=configs.model.miniblock,
                mode=configs.model.mode,
                v_max=configs.quantize.v_max,
                v_pi=configs.quantize.v_pi,
                in_bit=configs.quantize.in_bit,
                w_bit=configs.quantize.w_bit,
                photodetect=False,
                activation=False,
                act_thres=configs.model.act_thres
            )
        self.fc = TTM_Linear_module(
                in_features=d_v*n_head, 
                out_features=d_model, 
                bias=True, 
                shape=shape[1], 
                tensor_type='TensorTrainMatrix', 
                max_rank=rank[1],
                em_stepsize=em_stepsize,
                miniblock=configs.model.miniblock,
                mode=configs.model.mode,
                v_max=configs.quantize.v_max,
                v_pi=configs.quantize.v_pi,
                in_bit=configs.quantize.in_bit,
                w_bit=configs.quantize.w_bit,
                photodetect=False,
                activation=False,
                act_thres=configs.model.act_thres
            )
        
        # self.w_qs = TensorizedLinear_module(d_model, d_q*n_head, shape=shape[0], tensor_type=tensor_type,max_rank=rank[0],em_stepsize=em_stepsize,bias=True)
        # self.w_ks = TensorizedLinear_module(d_model, d_k*n_head, shape=shape[0], tensor_type=tensor_type,max_rank=rank[0],em_stepsize=em_stepsize,bias=True)
        # self.w_vs = TensorizedLinear_module(d_model, d_v*n_head, shape=shape[0], tensor_type=tensor_type,max_rank=rank[0],em_stepsize=em_stepsize,bias=True)

        # self.fc = TensorizedLinear_module(d_v*n_head, d_model, shape=shape[1], tensor_type=tensor_type,max_rank=rank[1],em_stepsize=em_stepsize,bias=True)


        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)


        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm((d_model,), eps=1e-12)

    

    def forward(self, q, k, v, mask=None, ranks=None,scale=None):


        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
    

        # q = self.w_qs(q,ranks,scale).view(sz_b, len_q, n_head, d_k)
        # k = self.w_ks(k,ranks,scale).view(sz_b, len_k, n_head, d_k)
        # v = self.w_vs(v,ranks,scale).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q,ranks,scale))
        
        q = self.dropout(self.fc(q))


        
        q += residual

        q = self.layer_norm(q)

        return q, attn

class Tensor_PFF(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], rank=[20,20], dropout=0.1,tensor_type = 'TensorTrain',
                bit_w = 8, scale_w = 2**(-5), 
                bit_a = 8, scale_a = 2**(-5), 
                quantized = False):
        super().__init__()
        
        em_stepsize = 1.0

       
        assert quantized == False
        
        self.fc_1 = TTM_Linear_module(
                in_features=d_in, 
                out_features=d_hid, 
                bias=True, 
                shape=shape[0], 
                tensor_type='TensorTrainMatrix', 
                max_rank=rank[0],
                em_stepsize=em_stepsize,
                miniblock=configs.model.miniblock,
                mode=configs.model.mode,
                v_max=configs.quantize.v_max,
                v_pi=configs.quantize.v_pi,
                in_bit=configs.quantize.in_bit,
                w_bit=configs.quantize.w_bit,
                photodetect=False,
                activation=False,
                act_thres=configs.model.act_thres
            )
        self.fc_2 = TTM_Linear_module(
                in_features=d_in, 
                out_features=d_hid, 
                bias=True, 
                shape=shape[1], 
                tensor_type='TensorTrainMatrix', 
                max_rank=rank[1],
                em_stepsize=em_stepsize,
                miniblock=configs.model.miniblock,
                mode=configs.model.mode,
                v_max=configs.quantize.v_max,
                v_pi=configs.quantize.v_pi,
                in_bit=configs.quantize.in_bit,
                w_bit=configs.quantize.w_bit,
                photodetect=False,
                activation=False,
                act_thres=configs.model.act_thres
            )
        # self.fc_1 = TensorizedLinear_module(d_in, d_hid, shape=shape[0], tensor_type=tensor_type,max_rank=rank[0],em_stepsize=em_stepsize)
        # self.fc_2 = TensorizedLinear_module(d_hid, d_in, shape=shape[1], tensor_type=tensor_type,max_rank=rank[1],em_stepsize=em_stepsize)

        # self.relu = nn.ReLU()
        self.relu = nn.GELU()

        self.layer_norm = nn.LayerNorm((d_in,), eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,ranks=None,scale=None):

        # print(torch.max(self.fc_1.tensor.factors[0]))

        residual = x

        # x = self.fc_1(x,ranks,scale)
        # x = self.relu(x)
        # x = self.fc_2(x,ranks,scale)
        
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)

        x = self.dropout(x)


        x = x + residual

        x = self.layer_norm(x)
        
        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))


  
        if mask!=None and len(mask.shape)==3:
            mask.to(torch.float)
            mask = mask.unsqueeze(1)
            mask = (1-mask)*(-1e9)
            # mask = torch.unsqueeze(mask,-1).to(torch.float)
            # mask = mask@ torch.transpose(mask,-1,-2)
        
        

        
        if mask is not None:
            attn = attn + mask
            # attn = attn.masked_fill(mask <= 0.1, -1e9)
            # attn = attn.masked_fill(mask <= 0.1, 0)
            # attn = torch.matmul(attn,mask)


        attn = F.softmax(attn, dim=-1)

        

        output = torch.matmul(self.dropout(attn), v)

        # print(output)

        return output, attn