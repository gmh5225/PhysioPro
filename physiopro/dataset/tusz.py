# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.fftpack import fft

from .forecast import DATASETS, ForecastDataset

def computeFFT(signals, n):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
        n: length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
    """
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0

    FT = np.log(amp)
    # P = np.angle(fourier_signal)

    return FT


@DATASETS.register_module("tusz")
class TUSZForecastDataset(ForecastDataset):
    def __init__(
            self,
            data_folder: Path,
            meta_path: Path,
            num_variables: int,
            freq: int = 256,
            time: int = 30,
            dataset: Optional[str] = "train",
            norm: bool = True,
            fft: bool = True,
            normalizer: Optional[Any] = None,
    ):
        super().__init__()
        self.num_nodes = num_variables
        self.freq = freq
        self.num_variables = freq // 2 * num_variables
        self.max_seq_len = time
        self.num_classes = 2
        self.norm = norm
        self.fft = fft

        meta = pd.read_pickle(meta_path)
        self.meta = meta
        feature_lines, label_lines = [], []
        for idx in meta[dataset]:
            feature_lines.append(pd.read_pickle(f"{data_folder}/feature_line/{idx}.pkl"))
            label_lines.append(pd.read_pickle(f"{data_folder}/label_line/{idx}.pkl"))
        feature = pd.concat(feature_lines).values.reshape(-1, self.num_nodes) # shape (105192000, 19) for tusz v1.5.2
        self.label = pd.concat(label_lines).values.astype(int) # shape (8876, 1) for tusz v1.5.2
        self.feature = feature.reshape(-1, self.max_seq_len, self.num_nodes * self.freq)

        assert len(self.feature) == len(self.label)
        self.label_index = np.arange(len(self.label))

    def __getitem__(self, index):
        assert self.feature is not None and self.label is not None
        feature = self.feature[index]  # time, nodes * freq
        if self.fft:
            feature = feature.reshape(-1, self.freq, self.num_nodes).transpose(0, 2, 1).reshape(-1, self.freq)
            feature = computeFFT(feature, self.freq).reshape(self.max_seq_len, self.num_nodes, -1)
        if self.norm:
            if self.fft:
                mean = "mean_fft@train"
                std = "std_fft@train"
            else:
                mean = "mean@train"
                std = "std@train"
            feature = (feature - self.meta[mean].values.reshape(-1, 1)) / self.meta[std].values.reshape(-1, 1)
        feature = feature.reshape(self.max_seq_len, -1)
        assert feature.shape[-1] == self.num_variables

        return feature.astype(np.float32), self.label[index]

    def freeup(self):
        ...

    def load(self):
        ...

    def downsample(self, rate: float):
        self.raw_feature = self.feature
        self.raw_label = self.label
        neg = np.argwhere(self.label == 0).flatten()
        pos = np.argwhere(self.label == 1).flatten()
        neg_size = len(neg)
        selected = np.random.choice(neg, int(neg_size * rate), replace=False)
        selected = np.concatenate([selected, pos])
        self.feature = self.feature[selected]
        self.label = self.label[selected]
        print(f"neg/pos ratio: {neg_size / self.label.sum():.2f} -> {len(selected) / self.label.sum():.2f}")

    def get_normalizer(self):
        return

    def restore(self):
        self.feature = self.raw_feature
        self.label = self.raw_label
