import torch.nn as nn
import torchaudio
import torch.nn.functional as F


class LogMelBase(nn.Module):
    def __init__(self,
                 sample_rate,
                 n_mels,
                 n_fft,
                 win_length,
                 hop_length,
                 f_min
                 ):
        super(LogMelBase, self).__init__()
        self.log_mel = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                 n_fft=n_fft,
                                                 win_length=win_length,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels,
                                                 f_min=f_min),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )

    def forward(self, x):
        """
        :param x: (B, N)
        :return: (B, F, T)
        """
        return self.log_mel(x)


if __name__ == '__main__':
    pass

