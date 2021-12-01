# -*- coding: utf-8 -*-
# @Time  : 2021/11/30 14:43
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : dataset.py
import argparse
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, args, subset):
        """
        Args:
            args: the path of L3DAS22 dataset
            subset: the subset of  dataset in [`L3DAS22_Task1_train100`, `L3DAS22_Task1_train360`, `L3DAS22_Task1_dev`]
        """
        self.args = args
        self.input_dir = args.input_dir
        self.data_dir = os.path.join(args.input_dir, subset, 'data')
        self.labels_dir = os.path.join(args.input_dir, subset, 'labels')
        self._load_files()

    def _pad(self, x, size: int):
        #pad all sounds to 4.792 seconds to meet the needs of Task1 baseline model MMUB
        length = x.shape[-1]
        if length > size:
            pad = x[:, :size]
        else:
            pad = np.zeros((x.shape[0], size))
            pad[:, :length] = x
        return pad

    def _load_files(self):
        files = os.listdir(self.data_dir)
        self.A_audios = [os.path.join(self.data_dir, i) for i in files if i.split('.')[0].split('_')[-1] == 'A']
        self.B_audios = [os.path.join(self.data_dir, i) for i in files if i.split('.')[0].split('_')[-1] == 'B']
        files = set([file.replace('_A', '').replace('_B', '')for file in files])
        self.target_audios = [os.path.join(self.labels_dir, i.replace('_A', '')) for i in files]
        self.target_labels = [os.path.join(self.labels_dir, i.replace('wav', 'txt')) for i in files]

    def __getitem__(self, item):
        # if both ambisonics mics are wanted
        sample_rate = self.args.sample_rate or 16000
        size = int(sample_rate * 4.792)
        samples, sr = librosa.load(self.A_audios[item], sample_rate, mono=False)

        if self.args.num_mics == 2:
            # stack the additional 4 channels to get a (8, samples) shap
            samples_B, sr = librosa.load(self.B_audios[item], sample_rate, mono=False)
            samples = np.concatenate((samples, samples_B), axis=-2)

        samples_target, sr = librosa.load(self.target_audios[item], sample_rate, mono=False)
        samples_target = samples_target.reshape((1, samples_target.shape[0]))

        samples = self._pad(samples, size)
        samples_target = self._pad(samples_target, size)

        return torch.Tensor(samples), torch.Tensor(samples_target)

    def __len__(self):
        return len(self.target_audios)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', required=True, help='the path of L3DAS22 dataset')
    parser.add_argument('--num_mics', default=1)
    parser.add_argument('--sample_rate', default=16000)

    args = parser.parse_args()

    audio = AudioDataset(args, 'L3DAS22_Task1_dev')

    for item in audio:
        pass