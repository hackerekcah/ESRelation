import torch.nn as nn
import torch.nn.functional as F
import torch
from layers.feature_base import LogMelBase


class LogMel(nn.Module):
    def __init__(self, sr=44100, n_mels=128, feat_interpolate=False):
        super(LogMel, self).__init__()
        self.n_mels = n_mels

        self.log_mel = LogMelBase(sample_rate=sr,
                                  n_fft=1024 if sr > 22050 else 512,  # 23.2ms
                                  win_length=1024 if sr > 22050 else 512,
                                  hop_length=512 if sr > 22050 else 256,  # 11.6ms
                                  n_mels=n_mels,
                                  f_min=0.)
        self.feat_interpolate = feat_interpolate

    def forward(self, x):
        # (B, C, T)
        x = self.log_mel(x)

        x = x.unsqueeze(1)

        if self.feat_interpolate:
            x = F.interpolate(x, size=(self.n_mels, 250))

        x = x.expand([-1, 3, -1, -1])

        return x


class LogMelWithVariousWinHop(nn.Module):
    def __init__(self, sr=44100, n_mels=128, n_frames=250, **kwargs):
        super(LogMelWithVariousWinHop, self).__init__()
        self.n_frames = n_frames
        self.n_mels = n_mels

        self.log_mel1 = LogMelBase(sample_rate=sr,
                                   n_fft=1024 if sr > 22050 else 512,
                                   win_length=1024 if sr > 22050 else 512,
                                   hop_length=512 if sr > 22050 else 256,
                                   n_mels=n_mels,
                                   f_min=0.)
        self.log_mel2 = LogMelBase(sample_rate=sr,
                                   n_fft=2048 if sr > 22050 else 1024,
                                   win_length=2048 if sr > 22050 else 1024,
                                   hop_length=1024 if sr > 22050 else 512,
                                   n_mels=n_mels,
                                   f_min=0.)
        self.log_mel3 = LogMelBase(sample_rate=sr,
                                   n_fft=4096 if sr > 22050 else 2048,
                                   win_length=4096 if sr > 22050 else 2048,
                                   hop_length=2048 if sr > 22050 else 1024,
                                   n_mels=n_mels,
                                   f_min=0.)

    def forward(self, x):
        feats = []
        # (B, C, T)
        x1 = self.log_mel1(x).unsqueeze(1)
        feats.append(F.interpolate(x1, size=(self.n_mels, self.n_frames)))
        x2 = self.log_mel2(x).unsqueeze(1)
        feats.append(F.interpolate(x2, size=(self.n_mels, self.n_frames)))
        x3 = self.log_mel3(x).unsqueeze(1)
        feats.append(F.interpolate(x3, size=(self.n_mels, self.n_frames)))
        x = torch.cat(feats, dim=1)

        return x


if __name__ == '__main__':
    pass