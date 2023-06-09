# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import h5py
import pyedflib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from scipy.fftpack import fft

from montage import INCLUDED_CHANNELS, bipolar


def get_tse_edf_file_names(raw_data_dir, resampled_data_dir):
    """
    Get .tse and .edf file names
    """
    tse_files = {}
    for path, _, files in os.walk(raw_data_dir):
        for name in files:
            if ".tse_bi" in name:
                assert name not in tse_files, "Duplicate tse_file"
                tse_files[name] = os.path.join(path, name)
    edf_files = {}
    for path, _, files in os.walk(resampled_data_dir):
        for name in files:
            if '.h5' in name:
                assert name not in edf_files, "Duplicate edf_file"
                edf_files[name] = os.path.join(path, name)
    return tse_files, edf_files


def extract_time(file_name):
    """
    Extract seizure time from .tse file

    The aggrement protocol defined for label annotation aggregation
    """
    seizure_times = []
    with open(file_name) as f:
        for line in f.readlines():
            if "seiz" in line:  # if seizure
                # seizure start and end time
                seizure_times.append(
                    [
                        float(line.strip().split(" ")[0]),
                        float(line.strip().split(" ")[1]),
                    ]
                )
    return seizure_times


def get_ordered_channels(file_name, verbose, labels_object, channel_names):
    """
    Some person may missing necessary channels, these persons will be excluded

    # refer to https://github.com/tsy935/eeg-gnn-ssl/blob/main/data/data_utils.py
    """
    labels = list(labels_object)
    for i, label in enumerate(labels):
        labels[i] = label.split("-")[0]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except:
            if verbose:
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels


def get_edf_signals(edf):
    """
    Get EEG signal in edf file

    Args:
    -----
        edf: edf object

    Returns:
    --------
        signals: shape (num_channels, num_data_points)
    """
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except:
            raise Exception("Get edf signals failed")
    return signals


def resample_data(signals, to_freq=200, window_size=4):
    """
    Resample signals from its original sampling freq to another freqency
    refer to https://github.com/tsy935/eeg-gnn-ssl/blob/main/data/resample_signals.py

    Args:
    -----
        signals: EEG signal slice, (num_channels, num_data_points)
        to_freq: Re-sampled frequency in Hz
        window_size: time window in seconds

    Returns:
    --------
        resampled: (num_channels, resampled_data_points)
    """
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)

    return resampled


def resample_all(raw_edf_dir, to_freq, save_dir):
    """
    Resample all edf files in raw_edf_dir to to_freq and save to save_dir
    """
    edf_files = []
    saved_files = []
    for path, _, files in os.walk(raw_edf_dir):
        for file in files:
            if ".edf" in file:
                edf_files.append(os.path.join(path, file))

    failed_files = []
    for idx in tqdm(range(len(edf_files))):
        edf_fn = edf_files[idx]

        save_fn = os.path.join(save_dir, edf_fn.split("/")[-1].split(".edf")[0] + ".h5")
        if os.path.exists(save_fn):
            saved_files.append(save_fn)
            continue
        try:
            f = pyedflib.EdfReader(edf_fn)

            ordered_channels = get_ordered_channels(
                edf_fn, False, f.getSignalLabels(), INCLUDED_CHANNELS
            )
            signals = get_edf_signals(f)
            signal_array = np.array(signals[ordered_channels, :])
            sample_freq = f.getSampleFrequency(0)
            if sample_freq != to_freq:
                signal_array = resample_data(
                    signal_array,
                    to_freq=to_freq,
                    window_size=int(signal_array.shape[1] / sample_freq),
                )

            with h5py.File(save_fn, "w") as hf:
                hf.create_dataset("resampled_signal", data=signal_array)
            saved_files.append(save_fn)

        except Exception:
            # pepole may missing some channels
            failed_files.append(edf_fn)

    print("DONE. {} files failed.".format(len(failed_files)))

    return saved_files


def get_train_test_ids(file_markers_dir, train_meta, test_meta):
    """
    Get train and test ids from file markers
    """
    def __get_ids(meta):
        ids = set()
        for line in open(meta, 'r').readlines():
            ids.add(line.split('.')[0].split('_')[0])
        return ids

    train_ids = [__get_ids(os.path.join(file_markers_dir, _)) for _ in train_meta]
    test_ids = [__get_ids(os.path.join(file_markers_dir, _)) for _ in test_meta]
    train_ids = train_ids[0] | train_ids[1]
    test_ids = test_ids[0] | test_ids[1]

    return train_ids, test_ids


def __get_feature_label_dataframe(f, edf_files, tse_files, frequency, duration, montage_type):
    """
    Get features and labels in dataframe format

    Returns:
    --------
    nedf: pandas dataframe, each row is a sample, each column is a channel
    label: pandas dataframe, each row is a sample, each column is a label
    """
    edf = np.array(h5py.File(edf_files[f], 'r')['resampled_signal'][()]).T

    assert len(edf) % frequency == 0, f"{f} EDF file cannot be reshape to 200 shape."
    num_samples = len(edf) // frequency  # in seconds

    time = extract_time(tse_files[f.split('.')[0] + '.tse_bi'])

    # generate labels in dataframe format
    labels = np.zeros((num_samples, 1))
    for clip_idx, _ in enumerate(labels):
        is_seizure = 0
        physical_clip_len = int(frequency)

        start_window = clip_idx * physical_clip_len
        end_window = start_window + physical_clip_len
        for t in time:
            start_t = int(t[0] * frequency)
            end_t = int(t[1] * frequency)
            if not ((end_window < start_t) or (start_window > end_t)):
                is_seizure = 1
                break
        labels[clip_idx][0] = is_seizure
    labels = pd.DataFrame(index=np.arange(len(labels)), data=labels)
    # truncate the additional labels
    labels = labels.iloc[: len(labels) - len(labels) % duration]

    # generate features in dataframe format
    features = pd.DataFrame(index=np.arange(len(edf)))
    channels_map = {k: v for v, k in enumerate(INCLUDED_CHANNELS)}
    if montage_type == 'bipolar':
        for bp in bipolar:
            start, end = bp.split('-')
            start = f"EEG {start}"
            end = f"EEG {end}"
            bp_value = edf[:, channels_map[start]] - edf[:, channels_map[end]]
            features.insert(len(features.columns), bp, bp_value)
    else:
        for bp in INCLUDED_CHANNELS:
            bp_value = edf[:, channels_map[bp]]
            features.insert(len(features.columns), bp, bp_value)

    features.insert(0, 'id', f)
    features.set_index('id', inplace=True)

    # truncate the additional seconds
    secs = int(len(features) / frequency)
    features = features.iloc[:(len(features) - secs % duration * frequency)]

    return features, labels


def get_features_and_labels(edf_files, tse_files, frequency, duration, montage_type, \
                             train_ids, test_ids, feature_dir, label_dir):
    """
    Get features and labels in dataframe format

    Returns:
    --------
    nedfs: dict, key is edf file name, value is pandas dataframe, each row is a sample, each column is a channel
    labels: dict, key is edf file name, value is pandas dataframe, each row is a sample, each column is a label
    """
    labels = {}
    nedfs = {}
    for edf_file in tqdm(edf_files):
        if os.path.exists(f"{feature_dir}/{edf_file}.pkl"):
            nedfs[edf_file] = pd.read_pickle(f"{feature_dir}/{edf_file}.pkl")
            labels[edf_file] = pd.read_pickle(f"{label_dir}/{edf_file}.pkl")
        else:
            idx = edf_file.split('.')[0].split('_')[0]
            if idx not in train_ids and idx not in test_ids:
                continue

            nedf, label = __get_feature_label_dataframe(edf_file, edf_files, tse_files, frequency, duration, montage_type)
            labels[edf_file] = label
            nedfs[edf_file] = nedf
            try:
                nedf.to_pickle(f"{feature_dir}/{edf_file}.pkl")
                label.to_pickle(f"{label_dir}/{edf_file}.pkl")
            except:
                raise Exception(f"{edf_file} IO error in serialization raw feature and label dataframe.")
    return nedfs, labels


def get_linewise_features_and_labels(edf_files, nedfs, labels, frequency, duration, train_ids, test_ids, \
                                     feature_linewise_dir, label_linewise_dir):
    """
    Calculate and serialize line-wise features and labels
    Format: One sample each line, DURATION * sampling_num * feature_num
    """
    for edf_file in tqdm(edf_files):
        if os.path.exists(f"{feature_linewise_dir}/{edf_file}.pkl"):
            nedf = pd.read_pickle(f"{feature_linewise_dir}/{edf_file}.pkl")
            label = pd.read_pickle(f"{label_linewise_dir}/{edf_file}.pkl")
        else:
            idx = edf_file.split('.')[0].split('_')[0]
            if idx not in train_ids and idx not in test_ids:
                continue

            nedf, label = nedfs[edf_file], labels[edf_file]
            nedf, label = nedf.copy(deep=True), label.copy(deep=True)
            sample_num = int(len(nedf) / frequency / duration)
            # sample_num * (duration * hz * feat_num)
            nedf = pd.DataFrame(index=nedf.index.values.reshape(-1, duration * frequency)[:, 0],
                                data=nedf.values.reshape(-1, duration * frequency * len(nedf.columns)),
                                columns=np.tile(np.array(nedf.columns), duration * frequency))
            label = pd.DataFrame(index=nedf.index,
                                 data=np.any(label.values.reshape(-1, duration), axis=1).astype(int))

            # Serialize to disk
            try:
                nedf.to_pickle(f"{feature_linewise_dir}/{edf_file}.pkl")
                label.to_pickle(f"{label_linewise_dir}/{edf_file}.pkl")
            except:
                raise Exception(f"{edf_file} IO error in serialization for LINED feature and label dataframe.")


def compute_fft(signals, n):
    """
    Args:
        signals: numpy.ndarray
            EEG signals, (number of channels, number of data points)
            shape (525960, 19, 200) for v1.5.2
        n: integer
            length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
    """
    # FFT on the last dimension
    fourier_signal = fft(signals, n=n, axis=-1)

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.] = 1e-8  # avoid log of 0

    FT = np.log(amp)
    # P = np.angle(fourier_signal) # risk of OOM

    return FT


def get_meta(montage_type, frequency, train_ids, test_ids, edf_files, nedfs, meta_dir):
    """
    Calculate and serialize meta information, including:
        channel-wise mean@train, std@train, mean@test, std@test, mean_fft@train, std_fft@train
    """
    edf_files_train = []
    edf_files_test = []
    for edf_file in edf_files:
        idx = edf_file.split('.')[0].split('_')[0]
        if idx in train_ids:
            edf_files_train.append(edf_file)
        if idx in test_ids:
            edf_files_test.append(edf_file)
    print(f'Number of train / test edf files: {len(edf_files_train)} / {len(edf_files_test)}')

    print("Calculating mean/std of train and test...")
    train_df = pd.concat([nedfs[i] for i in edf_files_train], axis='index')
    test_df = pd.concat([nedfs[i] for i in edf_files_test], axis='index')

    # mean, std of train and test
    mean_train, std_train = train_df.mean(axis='index'), train_df.std(axis='index')
    mean_test, std_test = test_df.mean(axis='index'), test_df.std(axis='index')

    print("Serializing train_df and test_df...")
    pd.to_pickle(train_df, f"{meta_dir}/train_df.pkl")
    pd.to_pickle(test_df, f"{meta_dir}/test_df.pkl")

    print("Calculating mean/std of train fft...")
    num_polars = len(bipolar) if montage_type == 'bipolar' else len(INCLUDED_CHANNELS)
    ffted = compute_fft(train_df.values.reshape(-1, frequency, num_polars).transpose(0, 2, 1), # second_num * num_channels * frequency
                        frequency)
    train_df_fft = pd.DataFrame(index=train_df.index,
                                columns=train_df.columns,
                                data=ffted.transpose(0, 2, 1).reshape(-1, num_polars))
    # mean, std of train fft
    mean_fft_train, std_fft_train = train_df_fft.mean(axis='index'), train_df_fft.std(axis='index')

    meta = {'train': edf_files_train,
            'test': edf_files_test,
            'mean@train': mean_train,
            'std@train': std_train,
            'mean@test': mean_test,
            'std@test': std_test,
            'mean_fft@train': mean_fft_train,
            'std_fft@train': std_fft_train}
    print(meta)

    print("Serializing meta...")
    pd.to_pickle(meta, f"{meta_dir}/meta.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tusz-version', type=str, default="2.0.0", choices=['1.5.2', '2.0.0'],
                        help='tusz version')
    parser.add_argument('--duration', type=int, default=60, choices=[30, 60, 120],
                        help='window size, in seconds')
    parser.add_argument('--frequency', type=int, default=200,
                        help='resample frequency')
    parser.add_argument('--montage-type', type=str, default="unipolar", choices=['unipolar', 'bipolar'],
                        help='unipolar or bipolar')
    parser.add_argument('--raw-data-dir', type=str, default="data/tuh_eeg_seizure",
                        help='path to raw data')
    parser.add_argument('--resampled-data-dir', type=str, default="data/tuh_eeg_seizure_resampled/",
                        help='path to resampled h5py files, the folder can be empty. If empty, the resampling process will be performed')
    parser.add_argument('--file-markers-dir', type=str, default="data/file_markers_detection",
                        help='path to file markers, only required in v1.5.2 for train/test splitting')
    parser.add_argument('--train-meta', type=str, nargs='+', default=['trainSet_seq2seq_12s_sz.txt', 'trainSet_seq2seq_12s_nosz.txt'],
                        help='file markers of train set, only required for v1.5.2.')
    parser.add_argument('--test-meta', type=str, nargs='+', default=['devSet_seq2seq_12s_nosz.txt', 'devSet_seq2seq_12s_sz.txt'],
                        help='file markers of train set, only required for v1.5.2.')
    parser.add_argument('--output-dir', type=str, default="data/tusz_processed/",
                        help='root path for output data, can be empty.')
    args = parser.parse_args()

    # output directories
    meta_dir = f"{args.output_dir}/{args.montage_type}/"  # meta
    feature_dir = f"{args.output_dir}/{args.montage_type}/feature/"  # features
    label_dir = f"{args.output_dir}/{args.montage_type}/label/"  # labels
    feature_linewise_dir = f"{args.output_dir}/{args.montage_type}/feature_line/"  # line-wise features
    label_linewise_dir = f"{args.output_dir}/{args.montage_type}/label_line/"  # line-wise labels

    print("Creating output directories...")
    for path in [args.output_dir, meta_dir, feature_dir, feature_linewise_dir, label_dir, label_linewise_dir, args.resampled_data_dir]:
        if not os.path.isdir(path):
            os.mkdir(path)

    print("Reading tse/edf file names...")
    tse_files, edf_files = get_tse_edf_file_names(args.raw_data_dir, args.resampled_data_dir)
    print(f"Number of tse files: {len(tse_files)}")
    print(f"Number of resampled edf files: {len(edf_files)}")

    # process resampleing if no files not found in resampled path
    print("Resampling...")
    if len(edf_files) == 0:
        edf_files = resample_all(args.raw_data_dir, args.frequency, args.resampled_data_dir)
    print(f"Number of resampled edf files: {len(edf_files)}")

    print("Getting train/test ids...")
    train_ids, test_ids = get_train_test_ids(args.file_markers_dir, args.train_meta, args.test_meta)
    print(f"Number of train ids: {len(train_ids)}")
    print(f"Number of test ids: {len(test_ids)}")

    print("Getting features and labels in dataframe...")
    nedfs, labels = get_features_and_labels(edf_files, tse_files, args.frequency, args.duration, args.montage_type, \
                                            train_ids, test_ids, feature_dir, label_dir)

    print("Getting line-wise features and labels...")
    get_linewise_features_and_labels(edf_files, nedfs, labels, args.frequency, args.duration, train_ids, test_ids, \
                                     feature_linewise_dir, label_linewise_dir)

    print("Getting meta...")
    get_meta(args.montage_type, args.frequency, train_ids, test_ids, edf_files, nedfs, meta_dir)

if __name__ == '__main__':
    main()
