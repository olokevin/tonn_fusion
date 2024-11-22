import pdb
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

from pyutils.config import configs
from model import LMF
from TOMFUN_model import TOMFUN

AUDIO = b'covarep'
VISUAL = b'facet'
TEXT = b'glove'
LABEL = b'label'
TRAIN = b'train'
VALID = b'valid'
TEST = b'test'

def build_fusion_model():
    if configs.model.name == 'LMF':
        model = LMF(
            input_dims = configs.model.input_dims, 
            hidden_dims = configs.model.hidden_dims, 
            sub_out_dims = configs.model.sub_out_dims, 
            dropouts = configs.model.dropouts, 
            output_dim = configs.model.output_dim, 
            rank = configs.model.rank, 

            FUSION_TYPE = configs.model.FUSION_TYPE,

            TT_ATTN = configs.model.TT_ATTN, 
            n_layers = configs.model.n_layers, 
            n_head = configs.model.n_head, 
            ATTN_rank = configs.model.ATTN_rank,

            TT_FUSION = configs.model.TT_FUSION, 
            TT_FUSION_rank = configs.model.TT_FUSION_rank, 
            tensor_type = configs.model.tensor_type,

            TT_SUBNET = configs.model.TT_SUBNET,
            TT_SUBNET_rank = configs.model.TT_SUBNET_rank,

            device = torch.device("cuda:" + str(configs.run.gpu_id)), 
            dtype = torch.float32, 
            use_softmax=False
        )
        return model
    elif configs.model.name == 'TOMFUN':
        model = TOMFUN(
            input_dims = configs.model.input_dims, 
            hidden_dims = configs.model.hidden_dims, 
            sub_out_dims = configs.model.sub_out_dims, 
            dropouts = configs.model.dropouts, 
            output_dim = configs.model.output_dim, 
            rank = configs.model.rank, 

            FUSION_TYPE = configs.model.FUSION_TYPE,

            TT_ATTN = configs.model.TT_ATTN, 
            n_layers = configs.model.n_layers, 
            n_head = configs.model.n_head, 
            ATTN_rank = configs.model.ATTN_rank,

            TT_FUSION = configs.model.TT_FUSION, 
            TT_FUSION_rank = configs.model.TT_FUSION_rank, 
            tensor_type = configs.model.tensor_type,

            TT_SUBNET = configs.model.TT_SUBNET,
            TT_SUBNET_rank = configs.model.TT_SUBNET_rank,

            device = torch.device("cuda:" + str(configs.run.gpu_id)), 
            dtype = torch.float32, 
            use_softmax=False
        )
        
        # inject non-ideality
        # deterministic phase bias
        if configs.noise.phase_bias:
            model.assign_random_phase_bias(random_state=int(configs.noise.random_state))
        # deterministic phase shifter gamma noise
        model.set_gamma_noise(
            float(configs.noise.gamma_noise_std), random_state=int(configs.noise.random_state)
        )
        # deterministic phase shifter crosstalk
        model.set_crosstalk_factor(float(configs.noise.crosstalk_factor))
        # deterministic phase quantization
        model.set_weight_bitwidth(int(configs.quantize.weight_bit))
        # enable/disable noisy identity
        model.set_noisy_identity(int(configs.sl.noisy_identity))
        
        return model
    else:
        raise ValueError("Model name not supported")

def build_fusion_dataloader():
    if configs.dataset.name == "iemocap":
        train_dataset, validation_dataset, test_set = load_iemocap(configs.dataset.dataset_dir, configs.dataset.emotion)
    else:
        raise ValueError("Dataset not supported")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.run.batch_size,
        shuffle=int(configs.dataset.shuffle),
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=configs.run.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=configs.run.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )
    return train_loader, validation_loader, test_loader

def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings


def load_iemocap(data_path, emotion):
    # parse the input args
    class IEMOCAP(Dataset):
        '''
        PyTorch Dataset for IEMOCAP, don't need to change this
        '''
        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    if sys.version_info.major == 2:
        iemocap_data = pickle.load(open(data_path + "iemocap.pkl", 'rb'))
    else:
        iemocap_data = pickle.load(open(data_path + "iemocap.pkl", 'rb'), encoding='bytes')

    keys_dict = list(iemocap_data.keys())
    if emotion=='angry':
        emotion=keys_dict[0]
    if emotion=='sad':
        emotion=keys_dict[1]
    if emotion=='neutral':
        emotion=keys_dict[2]
    if emotion=='happy':
        emotion=keys_dict[3]

    iemocap_train, iemocap_valid, iemocap_test = iemocap_data[emotion][TRAIN], iemocap_data[emotion][VALID], iemocap_data[emotion][TEST]

    train_audio, train_visual, train_text, train_labels \
        = iemocap_train[AUDIO], iemocap_train[VISUAL], iemocap_train[TEXT], iemocap_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = iemocap_valid[AUDIO], iemocap_valid[VISUAL], iemocap_valid[TEXT], iemocap_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = iemocap_test[AUDIO], iemocap_test[VISUAL], iemocap_test[TEXT], iemocap_test[LABEL]

    # code that instantiates the Dataset objects
    train_set = IEMOCAP(train_audio, train_visual, train_text, train_labels)
    valid_set = IEMOCAP(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = IEMOCAP(test_audio, test_visual, test_text, test_labels)


    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_seq = train_set[0][2].shape[0]
    print("Text feature sequence length is: {}".format(text_seq))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim, text_seq)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set
