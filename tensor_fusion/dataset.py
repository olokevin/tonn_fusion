import pickle
import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    '''
    Dataset for CMU-MOSI
    '''
    def __init__(self, text, audio, vision, labels):
        '''
        args:
            text: text modality feature of shape (N, seq. length, text_input_size)
            audio: audio modality feature of shape (N, seq. length, audio_input_size)
            vision: vision modality feature of shape (N, seq. length, vision_input_size)
            labels: labels of shape (N, 1) and ranges from -3 to 3
        '''
        self.text = text
        self.audio = audio
        self.vision = vision
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Returns an individual data composed of (text, audio, vision, label)

        Returns:
            text: text modality feature of shape (seq. length, text_input_size)
            audio: audio modality feature of shape (audio_input_size)
            vision: vision modality feature of shape (vision_input_size)
            label: a scalar label that ranges from -3 to 3
        '''
        text = self.text[idx]
        audio = self.audio[idx]
        vision = self.vision[idx]
        label = self.labels[idx]
        return text, audio, vision, label

class BinaryMultimodalDataset(Dataset):
    '''
    Dataset for CMU-MOSI
    '''
    def __init__(self, text, audio, vision, labels):
        '''
        args:
            text: text modality feature of shape (N, seq. length, text_input_size)
            audio: audio modality feature of shape (N, seq. length, audio_input_size)
            vision: vision modality feature of shape (N, seq. length, vision_input_size)
            labels: labels of shape (N, 1) and ranges from -3 to 3
        '''
        self.text = text
        self.audio = audio
        self.vision = vision
        self.labels = (labels > 0).type(labels.dtype)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Returns an individual data composed of (text, audio, vision, label)
        Returns:
            text: text modality feature of shape (seq. length, text_input_size)
            audio: audio modality feature of shape (audio_input_size)
            vision: vision modality feature of shape (vision_input_size)
            label: a scalar label that ranges from -3 to 3
        '''
        text = self.text[idx]
        audio = self.audio[idx]
        vision = self.vision[idx]
        label = self.labels[idx]
        return text, audio, vision, label

def get_cmu_mosi_dataset(binary=False, path='/home/christian_lee/projects/datasets/cmu-mosi/mosi_20_seq_data.pkl', device=None, dtype=None):
    '''
    args:
        binary: binary dataset if True regression dataset if False
        path: path to the dataset
        device:
        dtype:
    '''

    
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    
    # features: (batch_size, seq_length, input_size)
    # audio and vision features are averaged across time to (batch_size, input_size)

    # labels: (batch_size, 1)
    
    text = torch.tensor(data['train']['text'], device=device, dtype=dtype)
    audio = torch.tensor(data['train']['audio'], device=device, dtype=dtype).mean(dim=1)
    vision = torch.tensor(data['train']['vision'], device=device, dtype=dtype).mean(dim=1)
    labels = torch.tensor(data['train']['labels'], device=device, dtype=dtype).squeeze(1)
    if binary:
        train_set = BinaryMultimodalDataset(text, audio, vision, labels)
    else:
        train_set = MultimodalDataset(text, audio, vision, labels)

    text = torch.tensor(data['valid']['text'], device=device, dtype=dtype)
    audio = torch.tensor(data['valid']['audio'], device=device, dtype=dtype).mean(dim=1)
    vision = torch.tensor(data['valid']['vision'], device=device, dtype=dtype).mean(dim=1)
    labels = torch.tensor(data['valid']['labels'], device=device, dtype=dtype).squeeze(1)
    if binary:
        valid_set = BinaryMultimodalDataset(text, audio, vision, labels)
    else:    
        valid_set = MultimodalDataset(text, audio, vision, labels)
    
    text = torch.tensor(data['test']['text'], device=device, dtype=dtype)
    audio = torch.tensor(data['test']['audio'], device=device, dtype=dtype).mean(dim=1)
    vision = torch.tensor(data['test']['vision'], device=device, dtype=dtype).mean(dim=1)
    labels = torch.tensor(data['test']['labels'], device=device, dtype=dtype).squeeze(1)
    if binary:
        test_set = BinaryMultimodalDataset(text, audio, vision, labels)
    else:
        test_set = MultimodalDataset(text, audio, vision, labels)

    return train_set, valid_set, test_set