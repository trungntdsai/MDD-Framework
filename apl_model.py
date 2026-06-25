import torch
import torchaudio
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torch.nn.functional as F
import numpy as np

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

def init_sub_block(
    input_dim,
    model_dim,
    kernel_size=3,
    stride=2,
):
    return nn.Sequential(
        nn.Conv1d(input_dim, model_dim, kernel_size, stride, padding=1),
        nn.ReLU(),
    )


def calc_length(
    lengths, padding=1, kernel_size=3, stride=2, ceil_mode=False, repeat_num=1
):
    add_pad: float = (padding * 2) - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)

class PhoneCNNStack(nn.Module):
    def __init__(self):
        super().__init__()

        # self.fc = nn.Linear(1024,768)
        self.Conv2d = nn.Conv2d(1, 1, 3, 1, 1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.Conv2d(x)
        x = x.squeeze(1)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.reLU(x)
        x = self.drop_out(x)
        return x


class PhoneRNNStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(768)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, bidirectional=True, batch_first=True)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop_out(x)

        return x


class Phonetic_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = PhoneCNNStack()
        self.RNN = PhoneRNNStack()

    def forward(self, x):
        x = self.CNN(x)
        x = self.RNN(x)
        return x
    
class AcousticCNNStack1(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm1d(80)
        self.Conv2d = nn.Conv2d(1, 1, 3, 1, 1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.Conv2d(x)
        x = x.squeeze(1)
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)

        return x


class AcousticCNNStack(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm1d(80)
        self.Conv2d = nn.Conv2d(1, 1, 3, 1, 1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.Conv2d(x)
        x = x.squeeze(1)
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)

        return x


class AcousticRNNStack1(nn.Module):
    def __init__(self):
        super().__init__()

        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size=80, hidden_size=384, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop_out(x)
        return x


class AcousticRNNStack(nn.Module):
    def __init__(self):
        super().__init__()

        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop_out(x)
        return x


class Acoustic_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN1 = AcousticCNNStack1()
        self.CNN = AcousticCNNStack()
        self.RNN = AcousticRNNStack()
        self.RNN1 = AcousticRNNStack1()

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN(x)
        x = self.RNN1(x)
        x = self.RNN(x)
        return x
    
class Linguistic_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(256, 64)
        self.fc1 = nn.Linear(128, 2304)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, bidirectional=True, batch_first=True)
        self.fc3 = nn.Linear(128, 2304)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.long()
        x = self.embedding(x)
        o, _ = self.bilstm(x)
        y = self.fc1(o)
        x = self.fc3(o)
        return x, y
    
class APL(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.Acoustic_encoder = Acoustic_encoder()
        self.Phonetic_encoder = Phonetic_encoder()
        self.Linguistic_encoder = Linguistic_encoder()
        # self.text_to_tensor = text_to_tensor
        # self.tensor_to_text = tensor_to_text
        self.fc1 = nn.Linear(3072, 123, bias=True)
        self.multihead_attn = nn.MultiheadAttention(1536, 16, batch_first=True, kdim=2304, vdim=2304)
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, 
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def cal_mel(self, x):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
        return x

    def forward(self, waveform, input_values, linguistic):
        acoustic = self.cal_mel(waveform)
        phonetic = self.wav2vec2(input_values, attention_mask=None)[0]


        phonetic = self.Phonetic_encoder(phonetic)  # batch x time x 768
        acoustic = self.Acoustic_encoder(acoustic)  # batch x time x 768
        linguistic = self.Linguistic_encoder(linguistic)  # shape [0]: 2304 x len(canon)
        Hv = linguistic[0]
        Hk = linguistic[1]
        min_time = min(acoustic.size(1), phonetic.size(1))
        acoustic = acoustic[:, :min_time, :]
        phonetic = phonetic[:, :min_time, :]
        Hq = torch.cat((acoustic, phonetic), dim=2)
        attn_output, attn_output_weights = self.multihead_attn(Hq, Hk, Hv)
        c = attn_output
        before_Linear = torch.cat((c, Hq), dim=2)
        output = self.fc1(before_Linear)
        return output