import numpy as np
import torch
import torch.nn as nn

from .layers import TensorizedEmbedding, TensorizedLinear_module
from .Transformer_tensor_sublayers import DecoderLayer, EncoderLayer


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


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

class Encoder(nn.Module):
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
          tensor_shape = [[16,16,8,5],[4,4,8,4]]
          if tensorized:
              self.src_word_emb = TensorizedEmbedding(
                      tensor_type=emb_tensor_type,
                      max_rank=emb_rank,
                      shape=emb_shape,
                      prior_type='log_uniform',
                      eta=None)
          else:
              self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)




        # self.src_word_emb = model_origin._modules['distilbert']._modules['embeddings']
      
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        # self.position_enc = nn.Embedding(n_position,d_word_vec)
        self.token_type_emb = nn.Embedding(2,d_word_vec)

        self.register_buffer("position_ids", torch.arange(n_position).expand((1, -1)))

        self.dropout = nn.Dropout(p=dropout)


        if uncompressed==0:
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_model, d_inner, n_head,d_q, d_k, d_v, dropout=dropout,
                attention_shape=attention_shape,attention_rank=attention_rank,attention_tensor_type=attention_tensor_type,
                ffn_shape=ffn_shape,ffn_rank=ffn_rank,ffn_tensor_type=ffn_tensor_type,
                tensorized=tensorized,
                bit_attn = bit_attn, scale_attn = scale_attn, 
                bit_ffn = bit_ffn, scale_ffn = scale_ffn,
                bit_a = bit_a, scale_a = scale_a,
                quantized=quantized)
                for _ in range(n_layers)])
        else:
            self.layer_stack = [
                EncoderLayer(d_model, d_inner, n_head,d_q, d_k, d_v, dropout=dropout,
                attention_shape=attention_shape,attention_rank=attention_rank,attention_tensor_type=attention_tensor_type,
                ffn_shape=ffn_shape,ffn_rank=ffn_rank,ffn_tensor_type=ffn_tensor_type,
                tensorized=False,
                bit_attn = bit_attn, scale_attn = scale_attn, 
                bit_ffn = bit_ffn, scale_ffn = scale_ffn,
                bit_a = bit_a, scale_a = scale_a,
                quantized=quantized)
                for _ in range(uncompressed)]
            self.layer_stack += [EncoderLayer(d_model, d_inner, n_head,d_q, d_k, d_v, dropout=dropout,
                attention_shape=attention_shape,attention_rank=attention_rank,attention_tensor_type=attention_tensor_type,
                ffn_shape=ffn_shape,ffn_rank=ffn_rank,ffn_tensor_type=ffn_tensor_type,
                tensorized=tensorized,
                bit_attn = bit_attn, scale_attn = scale_attn, 
                bit_ffn = bit_ffn, scale_ffn = scale_ffn,
                bit_a = bit_a, scale_a = scale_a,
                quantized=quantized)
                for _ in range(n_layers-uncompressed)]
            self.layer_stack = nn.ModuleList(self.layer_stack)

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

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=True,
            trg_shape = [[16,16,8,5],[4,4,8,4]],trg_emb_rank = 16, trg_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True):

        super().__init__()
        
        if tensorized:
            self.trg_word_emb = TensorizedEmbedding(
                    tensor_type=trg_tensor_type,
                    max_rank=trg_emb_rank,
                    shape=trg_shape,
                    prior_type='log_uniform',
                    padding_idx=pad_idx,
                    eta=None)
        else:
            self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)


        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head,d_q, d_k, d_v, dropout=dropout,
            attention_shape=attention_shape,attention_rank=attention_rank,attention_tensor_type=attention_tensor_type,
            ffn_shape=ffn_shape,ffn_rank=ffn_rank,ffn_tensor_type=ffn_tensor_type,
            tensorized=tensorized,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output

############################### The attention-based subnetwork that is used in LMF for text #############################

class TextSubNet_attention(nn.Module):
    ''' The attention-based subnetwork that is used in LMF for text '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            d_classifier=512,num_class = 2, dropout_classifier = 0.2,
            classifier_shape = [4,4,8,4,4,4,8,4],classifier_rank = 20,classifier_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True):

        super().__init__()
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized)
        
        # self.preclassifier = nn.Linear(d_model,d_classifier)

        self.classifier = nn.Linear(d_classifier,num_class)

        if tensorized==True:
            self.preclassifier = TensorizedLinear_module(d_model, d_classifier, shape=classifier_shape, tensor_type=classifier_tensor_type,max_rank=classifier_rank)
        else:
            self.preclassifier = nn.Linear(d_model,d_classifier)

        self.dropout = nn.Dropout(p=dropout_classifier)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, src_seq, src_mask=None, return_attns=False):

        x = self.encoder(src_seq, src_mask, seg=None)
        # print(x)
        # x = torch.squeeze(x[:,0,:])
        x = torch.mean(x,1)
        # x = self.dropout(x)
        # x = self.preclassifier(x)
        x = self.classifier(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.dropout(x)
        # x = self.classifier(x)
        return x


class Transformer_translate(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab,n_trg_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            trg_shape = [[16,16,8,5],[4,4,8,4]],trg_emb_rank = 16, trg_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True):

        super().__init__()
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized)
        

        self.decoder = Decoder(n_trg_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            trg_shape = trg_shape,trg_emb_rank = trg_emb_rank, trg_tensor_type = trg_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized)
        
        if trg_tensor_type=='TensorTrain':
            decoder_shape = trg_shape[1]+trg_shape[0]
            decoder_rank = trg_emb_rank
            decoder_in = d_model
            decoder_out = np.prod(trg_shape[0])
            decoder_tensor_type = trg_tensor_type
        elif trg_tensor_type=='TensorTrainMatrix':
            decoder_shape = trg_shape[::-1]
            decoder_rank = trg_emb_rank
            decoder_in = d_model
            decoder_out = np.prod(trg_shape[0])
            decoder_tensor_type = trg_tensor_type


        if tensorized:
            self.trg_word_prj = TensorizedLinear_module(decoder_in, decoder_out, shape=decoder_shape, tensor_type=decoder_tensor_type,max_rank=decoder_rank)
        else:
            self.trg_word_prj = nn.Linear(d_model,n_trg_vocab)

    def forward(self, src_seq, trg_seq,src_mask=None,trg_mask=None):

        enc_output = self.encoder(src_seq,src_mask)
        dec_output = self.decoder(trg_seq, trg_mask, enc_output, src_mask)

        seq_logit = self.trg_word_prj(dec_output)
        
        return seq_logit


class Transformer_classifier(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            d_classifier=512,num_class = 2, dropout_classifier = 0.2,
            classifier_shape = [4,4,8,4,4,4,8,4],classifier_rank = 20,classifier_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True):

        super().__init__()
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized)
        
        # self.preclassifier = nn.Linear(d_model,d_classifier)

        self.classifier = nn.Linear(d_classifier,num_class)

        if tensorized==True:
            self.preclassifier = TensorizedLinear_module(d_model, d_classifier, shape=classifier_shape, tensor_type=classifier_tensor_type,max_rank=classifier_rank)
        else:
            self.preclassifier = nn.Linear(d_model,d_classifier)

        self.dropout = nn.Dropout(p=dropout_classifier)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, src_seq, src_mask, return_attns=False):

        x = self.encoder(src_seq,src_mask)
        # print(x)
        # x = torch.squeeze(x[:,0,:])
        x = torch.mean(x,1)
        # x = self.dropout(x)
        x = self.preclassifier(x)
        # x = self.relu(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x,x

class Transformer_Next(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            tensorized=True):

        super().__init__()
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,tensorized=tensorized)
        
        decoder_shape = emb_shape[::-1]
        decoder_rank = emb_rank
        decoder_in = d_model
        decoder_out = np.prod(decoder_shape[1])

        if tensorized:
            self.decoder = TensorizedLinear_module(decoder_in, decoder_out, shape=decoder_shape, tensor_type=emb_tensor_type,max_rank=decoder_rank)
        else:
            self.decoder = nn.Linear(d_model,n_src_vocab)

    def forward(self, src_seq, src_mask):

        x = self.encoder(src_seq,src_mask)
        x = self.decoder(x)

        return x



class Transformer_sentence(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab,d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            d_classifier=512,num_class = 3, dropout_classifier = 0.2,
            classifier_shape = [4,4,8,4,4,4,8,4],classifier_rank = 20,classifier_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True):

        super().__init__()
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized)
        if tensorized == True:
            self.preclassifier = TensorizedLinear_module(d_model*3, d_classifier, shape=classifier_shape, tensor_type=classifier_tensor_type,max_rank=classifier_rank)
        else:
            self.preclassifier = nn.Linear(d_model*3,d_classifier)
        self.classifier = nn.Sequential(self.preclassifier,nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_classifier,num_class))
        self.dropout = nn.Dropout(p=dropout)
        

  

    def forward(self, w1,w2,attn1=None,attn2=None):
        # device = self.classifier[0].weight.device



        output_1 = self.encoder(w1,attn1)
        output_2 = self.encoder(w2,attn2)

        #pooling
        # output_1 = torch.mean(output_1,1)
        # output_2 = torch.mean(output_2,1)

        output_1 = torch.squeeze(output_1[:,0,:])
        output_2 = torch.squeeze(output_2[:,0,:])

        diff = torch.abs(output_1-output_2)
        
        x = torch.cat((output_1,output_2,diff),dim=1)
        x = self.classifier(x)
        
        return x


class Transformer_sentence_concat(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab,d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            d_classifier=512,num_class = 3, dropout_classifier = 0.2,
            classifier_shape = [4,4,8,4,4,4,8,4],classifier_rank = 20,classifier_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True,
            uncompressed=0):

        super().__init__()
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized,
            uncompressed=uncompressed)

        if tensorized == True:
            self.preclassifier = TensorizedLinear_module(d_model, d_model, shape=classifier_shape, tensor_type=classifier_tensor_type,max_rank=classifier_rank)
        else:
            self.preclassifier = nn.Linear(d_model,d_model)
        
        # self.preclassifier = nn.Linear(d_model,d_model)

        self.classifier = nn.Sequential(self.preclassifier,nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,num_class))

        # self.classifier = nn.Sequential(nn.Linear(d_model,d_model),nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,num_class))


        

        

  

    def forward(self, w1, attn=None,seg=None,ranks=None,scale=None):
        # device = self.classifier[0].weight.device

        output = self.encoder(w1,attn,seg=seg,ranks=ranks,scale=scale)


        output = torch.squeeze(output[:,0,:])


        
        output = self.classifier(output)
        
        return output

from transformers import DistilBertConfig, DistilBertModel


class Transformer_sentence_pretrained(nn.Module):
    def __init__(
            self, d_classifier=768*3,num_class = 3, dropout_classifier = 0.2):

        super().__init__()

        d_model = 768
        
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.hid = nn.Linear(d_model,d_model)

        self.preclassifier = nn.Linear(d_model*3,d_classifier)
        self.classifier = nn.Sequential(self.preclassifier,nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_classifier,num_class))
        self.dropout = nn.Dropout(p=dropout_classifier)
        

  

    def forward(self, w1,w2):
        # device = self.classifier[0].weight.device



        output_1 = self.encoder(w1).last_hidden_state
        output_2 = self.encoder(w2).last_hidden_state

        #pooling
        output_1 = torch.mean(output_1,1)
        output_2 = torch.mean(output_2,1)

        # output_1 = torch.squeeze(output_1[:,0,:])
        # output_2 = torch.squeeze(output_2[:,0,:])



        diff = torch.abs(output_1-output_2)

        
        x = torch.cat((output_1,output_2,diff),dim=1)
        x = self.classifier(x)
        
        return x


class Transformer_sentence_pretrained_cat(nn.Module):
    def __init__(
            self, d_classifier=768,num_class = 2, dropout_classifier = 0.2):

        super().__init__()

        d_model = 768

        config = DistilBertConfig.from_pretrained('distilbert-base-uncased', 
                output_hidden_states=True, output_attentions=True)  
        
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased",config=config)

        self.hid = nn.Linear(d_model,d_model)

        self.preclassifier = nn.Linear(d_model,d_classifier)
        self.classifier = nn.Sequential(self.preclassifier,nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_classifier,num_class))
        
        self.dropout = nn.Dropout(p=dropout_classifier)
        

  

    def forward(self, w1,attn=None,seg=None):
        # device = self.classifier[0].weight.device



        output_1 = self.encoder(w1,attention_mask=attn).last_hidden_state


        #pooling
        # output_1 = torch.mean(output_1,1)


        output_1 = torch.squeeze(output_1[:,0,:])



        x = self.classifier(output_1)
        
        return x

from transformers import BertConfig, BertModel


class Transformer_sentence_BERT_cat(nn.Module):
    def __init__(
            self, d_classifier=768,num_class = 3, dropout_classifier = 0.2):

        super().__init__()

        d_model = 768

        config = BertConfig.from_pretrained('bert-base-uncased', 
                output_hidden_states=True, output_attentions=True)  
        
        self.encoder = BertModel.from_pretrained("bert-base-uncased",config=config)

        self.hid = nn.Linear(d_model,d_model)

        self.preclassifier = nn.Linear(d_model,d_classifier)
        # self.classifier = nn.Sequential(self.preclassifier,nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_classifier,num_class))
        self.classifier = nn.Linear(d_classifier,num_class)
        self.dropout = nn.Dropout(p=dropout_classifier)
        

  

    def forward(self, w1,attn=None,seg=None):
        # device = self.classifier[0].weight.device

    

        output_1 = self.encoder(w1,attention_mask=attn,token_type_ids=seg).pooler_output

        

        output_1 = self.dropout(output_1)



        x = self.classifier(output_1)

        
        return x



class BERT_tensor(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab,d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            d_classifier=512,num_class = 3, dropout_classifier = 0.2,
            classifier_shape = [4,4,8,4,4,4,8,4],classifier_rank = 20,classifier_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True,
            uncompressed=0):

        super().__init__()
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized,
            uncompressed=uncompressed)

        if tensorized == True:
            self.preclassifier = TensorizedLinear_module(d_model, d_model, shape=classifier_shape, tensor_type=classifier_tensor_type,max_rank=classifier_rank)
        else:
            self.preclassifier = nn.Linear(d_model,d_model)
        
        # self.preclassifier = nn.Linear(d_model,d_model)

        self.classifier = nn.Sequential(self.preclassifier,nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,num_class))

        self.token_classifier = nn.Sequential(nn.Linear(d_model,d_model),nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,30522))

        # self.token_classifier = nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.LayerNorm(d_model),nn.Linear(d_model,30522))

        # self.classifier = nn.Sequential(nn.Linear(d_model,d_model),nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,num_class))


        

        

  

    def forward(self, w1, attn=None,seg=None):
        # device = self.classifier[0].weight.device

        output = self.encoder(w1,attn,seg=seg)

        

        # output = torch.squeeze(output[:,0,:])

        # print(output.shape)
        
        # output = self.classifier(output)
    
        return self.classifier(output[:,0,:]), self.token_classifier(output)

class BERT_tensor_new(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab,d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            d_classifier=512,num_class = 3, dropout_classifier = 0.2,
            classifier_shape = [4,4,8,4,4,4,8,4],classifier_rank = 20,classifier_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True,
            uncompressed=0):

        super().__init__()

        
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized,
            uncompressed=uncompressed)

        if tensorized == True:
            self.preclassifier = TensorizedLinear_module(d_model, d_model, shape=classifier_shape, tensor_type=classifier_tensor_type,max_rank=classifier_rank)
        else:
            self.preclassifier = nn.Linear(d_model,d_model)
        
        # self.preclassifier = nn.Linear(d_model,d_model)

        self.classifier = nn.Sequential(self.preclassifier,nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,num_class))

        # self.token_classifier = nn.Sequential(nn.Linear(d_model,d_model),nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,30522))

        self.token_classifier = nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.LayerNorm(d_model),nn.Linear(d_model,30522))

        # self.classifier = nn.Sequential(nn.Linear(d_model,d_model),nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,num_class))


        

        

  

    def forward(self, w1, attn=None,seg=None):
        # device = self.classifier[0].weight.device

        output = self.encoder(w1,attn,seg=seg)

        

        # output = torch.squeeze(output[:,0,:])

        # print(output.shape)
        
        # output = self.classifier(output)
    
        return self.classifier(output[:,0,:]), self.token_classifier(output)



class Transformer_sentence_concat_module_replace(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab,d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            d_classifier=512,num_class = 3, dropout_classifier = 0.2,
            classifier_shape = [4,4,8,4,4,4,8,4],classifier_rank = 20,classifier_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True,
            uncompressed=0,
            BERT = None):

        super().__init__()

        self.BERT = BERT

        for p in BERT.parameters():
            p.requires_grad = False
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized,
            uncompressed=uncompressed)

        if tensorized == True:
            self.preclassifier = nn.Linear(d_model,d_model)
            # self.preclassifier = TensorizedLinear_module(d_model, d_model, shape=classifier_shape, tensor_type=classifier_tensor_type,max_rank=classifier_rank)
        else:
            self.preclassifier = nn.Linear(d_model,d_model)
        
        # self.preclassifier = nn.Linear(d_model,d_model)

        self.classifier = nn.Sequential(self.preclassifier,nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,num_class))

        # self.classifier = nn.Sequential(nn.Linear(d_model,d_model),nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,num_class))


        

        

  

    def forward(self, w1, attn=None,seg=None,p=0.1):
        # device = self.classifier[0].weight.device
        


        enc_output = self.encoder.src_word_emb(w1)
        
        # if self.scale_emb:
        #     enc_output *= self.d_model ** 0.5
        position_ids = self.encoder.position_ids[:, 0: w1.shape[1]]

        if seg==None:
            seg = torch.zeros(attn.shape).to(int).to(attn.device)

        # print(position_ids)
        
        # enc_output = self.dropout(self.position_enc(enc_output))

        enc_output = enc_output+self.encoder.position_enc(position_ids)+self.encoder.token_type_emb(seg)

        enc_output = self.encoder.layer_norm(enc_output)
        enc_output = self.encoder.dropout(enc_output)

        out_student = enc_output
        out_teacher = self.BERT.encoder.embeddings(w1,position_ids=position_ids,token_type_ids=seg)

        prob = int(torch.rand(1)<p)
        out = prob*out_student + (1-prob)*out_teacher

        extended_attention_mask = attn[:, None, None, :]

        for i in range(6):
            out_student = self.encoder.layer_stack[i](out, slf_attn_mask=attn)[0]
            out_teacher = self.BERT.encoder.encoder.layer[2*i](out,attention_mask=extended_attention_mask)[0]
            out_teacher = self.BERT.encoder.encoder.layer[2*i+1](out_teacher,attention_mask=extended_attention_mask)[0]

            prob = int(torch.rand(1)<p)
            out = prob*out_student + (1-prob)*out_teacher


        # output = self.encoder(w1,attn,seg=seg)


        out = torch.squeeze(out[:,0,:])


        
        output = self.classifier(out)
        
        return output