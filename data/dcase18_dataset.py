import os
import pandas as pd
from sklearn import preprocessing
import numpy as np
import soundfile as sf
import resampy
import torch
from data import register_dataset

_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


@register_dataset
class Dcase18TaskADevSet(torch.utils.data.Dataset):
    """
    sr=48000, 10seconds, dual-channels, 10 classes.
    """
    def __init__(self, fold=1, split='train', target_sr=48000, transform=None,
                 spl_norm=False,
                 random_channel=False):
        """
        :param split: 'train', 'evaluate'
        :param transform: callable
        """
        super(Dcase18TaskADevSet, self).__init__()

        self.dev_path = "/data/songhongwei/dcase2018_baseline/task1/datasets/TUT-urban-acoustic-scenes-2018-development"
        if split == 'train':
            csv_file = os.path.join(self.dev_path, 'evaluation_setup', 'fold1_{}.txt'.format(split))
        else:
            csv_file = os.path.join(self.dev_path, 'evaluation_setup', 'fold1_{}.txt'.format('evaluate'))
        df = pd.read_csv(csv_file, sep='\t', names=['file_names', 'labels'])

        self.file_names = df['file_names'].to_list()
        self.labels = df['labels'].to_list()

        if target_sr == 44100:
            prefix = 'audio44100'
            self.file_names = [fname.replace('audio', prefix) for fname in self.file_names]
        elif target_sr == 22050:
            prefix = 'audio22050'
            self.file_names = [fname.replace('audio', prefix) for fname in self.file_names]

        le = preprocessing.LabelBinarizer()
        self.labels = le.fit_transform(np.array(self.labels))

        self.transform = transform
        self.spl_norm = spl_norm
        self.target_sr = target_sr
        self.random_channel = random_channel

    def get_train_labels(self):
        # self.labels is binary label, return int label
        return np.argmax(self.labels, axis=1)

    def __getitem__(self, index):
        """
        :param index (int):
        :return:
            audio (torch.float32), shape=(N,)
            label (torch.float32), shape=(nb_classes,)
        """
        file_path = os.path.join(self.dev_path, self.file_names[index])

        # (N, C), audio is not normalized ot [-1, 1] by default
        audio, sr = sf.read(file_path)

        if audio.ndim == 2:
            if self.random_channel:
                idx = np.random.randint(0, 3)
                if idx == 2:
                    audio = np.mean(audio, 1)
                else:
                    audio = audio[:, idx]
            else:
                # mean channel
                audio = np.mean(audio, 1)

        # Resample if necessary
        if sr != self.target_sr:
            # (N,)
            audio = resampy.resample(audio, sr_orig=sr, sr_new=self.target_sr, filter='kaiser_best')

        if self.spl_norm:
            max_abs = max(abs(audio))
            # avoid divide by zero
            if (max_abs - 0.0) > 1e-12:
                audio = audio / max_abs

        binary_label = self.labels[index]
        label = np.argmax(binary_label)
        # must to torch.float32 for fft
        sample = (torch.as_tensor(audio, dtype=torch.float32),
                  torch.as_tensor(label, dtype=torch.int64))

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.file_names)


@register_dataset
class Dcase18TaskADevSetDual(torch.utils.data.Dataset):
    """
    sr=48000, 10seconds, dual-channels, 10 classes.
    """
    def __init__(self, fold=1, split='train', target_sr=48000, transform=None, spl_norm=False):
        """
        :param split: 'train', 'evaluate'
        :param transform: callable
        """
        super(Dcase18TaskADevSetDual, self).__init__()

        self.dev_path = "/data/songhongwei/dcase2018_baseline/task1/datasets/TUT-urban-acoustic-scenes-2018-development"
        if split == 'train':
            csv_file = os.path.join(self.dev_path, 'evaluation_setup', 'fold1_{}.txt'.format(split))
        else:
            csv_file = os.path.join(self.dev_path, 'evaluation_setup', 'fold1_{}.txt'.format('evaluate'))
        df = pd.read_csv(csv_file, sep='\t', names=['file_names', 'labels'])

        self.file_names = df['file_names'].to_list()
        self.labels = df['labels'].to_list()

        if target_sr == 44100:
            prefix = 'audio44100'
            self.file_names = [fname.replace('audio', prefix) for fname in self.file_names]
        elif target_sr == 22050:
            prefix = 'audio22050'
            self.file_names = [fname.replace('audio', prefix) for fname in self.file_names]

        le = preprocessing.LabelBinarizer()
        self.labels = le.fit_transform(np.array(self.labels))

        self.transform = transform
        self.spl_norm = spl_norm
        self.target_sr = target_sr

    def get_train_labels(self):
        # self.labels is binary label, return int label
        return np.argmax(self.labels, axis=1)

    def __getitem__(self, index):
        """
        :param index (int):
        :return:
            audio (torch.float32), shape=(N,)
            label (torch.float32), shape=(nb_classes,)
        """
        file_path = os.path.join(self.dev_path, self.file_names[index])

        # (N, C), audio is not normalized ot [-1, 1] by default
        audio, sr = sf.read(file_path)

        if audio.ndim == 2:
            # concatenate if multichannel
            audio = np.concatenate((audio[:, 0], audio[:, 1]))

        # Resample if necessary
        if sr != self.target_sr:
            # (N,)
            audio = resampy.resample(audio, sr_orig=sr, sr_new=self.target_sr, filter='kaiser_best')

        if self.spl_norm:
            max_abs = max(abs(audio))
            # avoid divide by zero
            if (max_abs - 0.0) > 1e-12:
                audio = audio / max_abs

        binary_label = self.labels[index]
        label = np.argmax(binary_label)
        # must to torch.float32 for fft
        sample = (torch.as_tensor(audio, dtype=torch.float32),
                  torch.as_tensor(label, dtype=torch.int64))

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.file_names)


if __name__ == '__main__':
    ds = Dcase18TaskADevSet(split='train', target_sr=48000, spl_norm=False,
                            random_channel=True)
    for d in ds:
        pass