# -*- coding: utf-8 -*-
# @Time  : 2021/12/2 10:07
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : Cunet.py
import torch
import torch.nn as nn
from einops import rearrange


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        return self.conv(x)


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        return self.conv(x)


class CUNET(nn.Module):
    """
    Input: [B, C, F, T]
    Output: [B, C, T, F]
    """
    def __init__(self,
                 fft_size=512,
                 hop_size=128,
                 ):
        super(CUNET, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = fft_size
        self.valid_freq = int(self.fft_size / 2)
        self.conv_block_1 = CausalConvBlock(4, 32, (6, 2), (2, 1))
        self.conv_block_2 = CausalConvBlock(32, 32, (6, 2), (2, 1))
        self.conv_block_3 = CausalConvBlock(32, 64, (7, 2), (2, 1))
        self.conv_block_4 = CausalConvBlock(64, 64, (6, 2), (2, 1))
        self.conv_block_5 = CausalConvBlock(64, 96, (6, 2), (2, 1))
        self.conv_block_6 = CausalConvBlock(96, 96, (6, 2), (2, 1))
        self.conv_block_7 = CausalConvBlock(96, 128, (2, 2), (2, 1))
        self.conv_block_8 = CausalConvBlock(128, 256, (2, 2), (1, 1))

        self.tran_conv_block_1 = CausalTransConvBlock(256, 256, (2, 2), (1, 1))
        self.tran_conv_block_2 = CausalTransConvBlock(256 + 128, 128, (2, 2), (2, 1))
        self.tran_conv_block_3 = CausalTransConvBlock(128 + 96, 96, (6, 2), (2, 1))
        self.tran_conv_block_4 = CausalTransConvBlock(96 + 96, 96, (6, 2), (2, 1))
        self.tran_conv_block_5 = CausalTransConvBlock(96 + 64, 64, (6, 2), (2, 1))
        self.tran_conv_block_6 = CausalTransConvBlock(64 + 64, 64, (7, 2), (2, 1))
        self.tran_conv_block_7 = CausalTransConvBlock(64 + 32, 32,  (6, 2), (2, 1))
        self.tran_conv_block_8 = CausalTransConvBlock(32 + 32, 4,  (6, 2), (2, 1))
        self.dense = nn.Linear(512, 512)


    def extract_features(self, inputs, device):
        # shape: [B, C, S]
        batch_size, channel, samples = inputs.size()

        features = []
        for idx in range(batch_size):
            # shape: [C, F, T, 2]
            features_batch = torch.stft(
                inputs[idx, ...],
                self.fft_size,
                self.hop_size,
                self.win_size,
                torch.hann_window(self.win_size).to(device),
                pad_mode='constant',
                onesided=True,
                return_complex=False)
            features.append(features_batch)

        # shape: [B, C, F, T, 2]
        features = torch.stack(features, 0)
        features = features[:, :, :self.valid_freq, :, :]
        real_features = features[..., 0]
        imag_features = features[..., 1]

        return real_features, imag_features

    def forward(self, x, device):
        # shape: [B, C, F, T]
        real_features, imag_features = self.extract_features(x, device)
        # shape: [B, C, F*2, T]
        x = torch.cat((real_features, imag_features), 2)

        e1 = self.conv_block_1(x)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = self.conv_block_6(e5)
        e7 = self.conv_block_7(e6)
        e8 = self.conv_block_8(e7)

        d1 = self.tran_conv_block_1(e8)
        d2 = self.tran_conv_block_2(torch.cat((d1, e7), 1))
        d3 = self.tran_conv_block_3(torch.cat((d2, e6), 1))
        d4 = self.tran_conv_block_4(torch.cat((d3, e5), 1))
        d5 = self.tran_conv_block_5(torch.cat((d4, e4), 1))
        d6 = self.tran_conv_block_6(torch.cat((d5, e3), 1))
        d7 = self.tran_conv_block_7(torch.cat((d6, e2), 1))
        out = self.tran_conv_block_8(torch.cat((d7, e1), 1))
        out = out.permute(0, 1, 3, 2)
        out = self.dense(out)
        out = rearrange(out, 'B C T F -> B C F T')

        real_mask = out[:, :, :self.valid_freq, :]
        imag_mask = out[:, :, self.valid_freq:, :]

        est_speech_real = torch.mul(real_features, real_mask) - torch.mul(imag_features, imag_mask)
        est_speech_imag = torch.mul(real_features, imag_mask) + torch.mul(imag_features, real_mask)
        est_speech_stft = torch.complex(est_speech_real, est_speech_imag)

        # shape: [B, C, F, T]
        est_speech_stft = torch.sum(est_speech_stft, 1)
        batch_size, frequency, frame = est_speech_stft.size()
        est_speech_stft = torch.cat((est_speech_stft, torch.zeros(batch_size, 1, frame).to(device)), 1)

        # shape: [B, S]
        est_speech = torch.istft(
            est_speech_stft,
            self.fft_size,
            self.hop_size,
            self.win_size,
            torch.hann_window(self.win_size).to(device))
        # shape: [B, 1, S]
        return torch.unsqueeze(est_speech, 1)


if __name__ == '__main__':
    layer = CUNET()
    x = torch.rand(1, 8, 512, 249)
    print("input shape:", x.shape)
    print("output shape:", layer(x).shape)
    l2 = nn.MSELoss()
    print(f"loss : {l2(x, layer(x))}")
    total_num = sum(p.numel() for p in layer.parameters())
    print(total_num)