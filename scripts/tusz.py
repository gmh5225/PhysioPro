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


def get_edf_file_names(resampled_data_dir):
    """
    Get .edf file names

    Returns:
    --------
    dict
        edf file names and corresponding paths
        file names are in format {official patient number}_{session number}_{segment number}, e.g., 00000258_s002_t000
    """
    edf_files = {}
    for root, _, files in os.walk(resampled_data_dir):
        for name in files:
            if '.h5' in name:
                # some patients have sessions listed under multiple montage folders, only the last one will be saved
                edf_files[name] = os.path.join(root, name)

    return edf_files


def get_label_file_names(version, raw_data_dir):
    """
    Get label file names

    Returns:
    --------
    dict
        label file names and corresponding paths
    """
    # postfix of label files
    postfix = 'tse_bi' if version == '1.5.2' else 'csv_bi'
    label_files = {}
    for root, _, files in os.walk(raw_data_dir):
        for name in files:
            if postfix in name:
                # some patients have sessions listed under multiple montage folders, only the last one will be saved
                label_files[name] = os.path.join(root, name)

    return label_files


def get_ordered_channels(file_name, verbose, labels_object, channel_names):
    """
    Some person may missing necessary channels, these persons will be excluded
    refer to https://github.com/tsy935/eeg-gnn-ssl/blob/main/data/data_utils.py
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


def get_edf_signals(edf, ordered_channels):
    """
    Get EEG signal in edf file

    Parameters:
    -----------
    edf:
        edf object
    ordered_channels: list
        list of channel indexes

    Returns:
    --------
    numpy.ndarray
        shape (num_channels, num_data_points)
    """
    signals = np.zeros((len(ordered_channels), edf.getNSamples()[0]))
    for i, index in enumerate(ordered_channels):
        try:
            signals[i, :] = edf.readSignal(index)
        except:
            raise Exception("Get edf signals failed")
    return signals


def resample_data(signals, to_freq=200, window_size=4):
    """
    Resample signals from its original sampling freq to another freqency
    refer to https://github.com/tsy935/eeg-gnn-ssl/blob/main/data/resample_signals.py

    Parameters:
    -----------
    signals: numpy.ndarray
        EEG signal slice, (num_channels, num_data_points)
    to_freq: int
        Re-sampled frequency in Hz
    window_size: int
        time window in seconds

    Returns:
    --------
    numpy.ndarray
        shape (num_channels, resampled_data_points)
    """
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)

    return resampled


def resample_all(raw_edf_dir, to_freq, save_dir):
    """
    Resample all edf files in raw_edf_dir to to_freq and save to save_dir

    Returns:
    --------
    dict
        edf file names and corresponding paths after resampling
    """
    raw_edfs = []
    for root, _, files in os.walk(raw_edf_dir):
        for file in files:
            if ".edf" in file:
                raw_edfs.append(os.path.join(root, file))
    print(f"Number of raw edf files: {len(raw_edfs)}")

    resampled_edfs = {}
    failed_files = []
    for _, edf_fn in enumerate(tqdm(raw_edfs)):
        new_file_name = f"{edf_fn.split('/')[-1].split('.edf')[0]}.h5"
        resampled_edf = os.path.join(save_dir, new_file_name)
        if os.path.exists(resampled_edf):
            resampled_edfs[new_file_name] = resampled_edf
            continue
        try:
            f = pyedflib.EdfReader(edf_fn)

            ordered_channels = get_ordered_channels(
                edf_fn, False, f.getSignalLabels(), INCLUDED_CHANNELS
            )
            signal_array = get_edf_signals(f, ordered_channels)
            sample_freq = f.getSampleFrequency(0)
            if sample_freq != to_freq:
                signal_array = resample_data(
                    signal_array,
                    to_freq=to_freq,
                    window_size=int(signal_array.shape[1] / sample_freq),
                )

            with h5py.File(resampled_edf, "w") as hf:
                hf.create_dataset("resampled_signal", data=signal_array)
            resampled_edfs[new_file_name] = resampled_edf

        except Exception:
            # pepole may missing some channels
            failed_files.append(edf_fn)

    print("DONE. {} files failed.".format(len(failed_files)))

    return resampled_edfs


def get_train_test_ids_v1(file_markers_dir, train_meta, test_meta):
    """
    Get train and test ids from file markers in v1.5.2

    Returns:
    --------
    list
        official patient numbers of train set
    list
        official patient numbers of test set
    """
    def __get_ids(meta):
        ids = set()
        for line in open(meta, 'r').readlines():
            ids.add(line.split('.')[0].split('_')[0])
        return ids

    train_ids, test_ids = set(), set()
    for meta in train_meta:
        train_ids.update(__get_ids(os.path.join(file_markers_dir, meta)))
    for meta in test_meta:
        test_ids.update(__get_ids(os.path.join(file_markers_dir, meta)))

    return list(train_ids), list(test_ids)


def get_train_test_ids_v2(raw_data_dir):
    """
    Get train and test ids in v2.0.0

    Returns:
    --------
    list
        official patient numbers of train set
    list
        official patient numbers of test set
    """
    train_ids = os.listdir(os.path.join(raw_data_dir, 'edf', 'train'))
    test_ids = os.listdir(os.path.join(raw_data_dir, 'edf', 'eval'))

    return train_ids, test_ids


def extract_seizure_time(version, file_name):
    """
    Extract seizure time from annnotation file

    Returns:
    --------
    list
        seizure time, each element is a list of start and end time
    """
    seizure_times = []
    with open(file_name) as f:
        for line in f.readlines():
            if "seiz" in line:  # if seizure
                # seizure start and end time
                if version == '1.5.2':
                    seizure_times.append([
                        float(line.strip().split(" ")[0]),
                        float(line.strip().split(" ")[1]),
                    ])
                else:
                    seizure_times.append([
                        float(line.strip().split(",")[1]),
                        float(line.strip().split(",")[2]),
                    ])
    return seizure_times


def __get_feature_label_dataframe(version, edf_file, edf_files, label_files, frequency, duration, montage_type):
    """
    Get features and labels in dataframe format

    Returns:
    --------
    pandas dataframe
        feature, each row is a sample, each column is a channel
    pandas dataframe
        label, each row is a sample, each column is a label
    """
    edf = np.array(h5py.File(edf_files[edf_file], 'r')['resampled_signal'][()]).T
    assert len(edf) % frequency == 0, f"{edf_file} EDF file shape error."
    num_samples = len(edf) // frequency  # in seconds

    postfix = '.tse_bi' if version == '1.5.2' else '.csv_bi'
    time = extract_seizure_time(version, label_files[edf_file.split('.')[0] + postfix])

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

    features.insert(0, 'id', edf_file)
    features.set_index('id', inplace=True)

    # truncate the additional seconds
    secs = int(len(features) / frequency)
    features = features.iloc[:(len(features) - secs % duration * frequency)]

    return features, labels


def get_features_and_labels(version, edf_files, label_files, frequency, duration, montage_type,
                            train_ids, test_ids, feature_dir, label_dir):
    """
    Get features and labels in dataframe format

    Returns:
    --------
    dict
        features, key is edf file name, value is pandas dataframe, each row is a sample, each column is a channel
    dict
        labels, key is edf file name, value is pandas dataframe, each row is a sample, each column is a label
    """
    nedfs = {}
    labels = {}
    for edf_file in tqdm(edf_files):
        if os.path.exists(f"{feature_dir}/{edf_file}.pkl"):
            nedfs[edf_file] = pd.read_pickle(f"{feature_dir}/{edf_file}.pkl")
            labels[edf_file] = pd.read_pickle(f"{label_dir}/{edf_file}.pkl")
        else:
            idx = edf_file.split('.')[0].split('_')[0]
            if idx not in train_ids and idx not in test_ids:
                continue

            nedf, label = __get_feature_label_dataframe(version, edf_file, edf_files, label_files, frequency, duration, montage_type)
            labels[edf_file] = label
            nedfs[edf_file] = nedf
            try:
                nedf.to_pickle(f"{feature_dir}/{edf_file}.pkl")
                label.to_pickle(f"{label_dir}/{edf_file}.pkl")
            except:
                raise Exception(f"{edf_file} IO error in serialization raw feature and label dataframe.")
    return nedfs, labels


def get_linewise_features_and_labels(edf_files, nedfs, labels, frequency, duration, train_ids, test_ids,
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
    Parameters:
    -----------
    signals: numpy.ndarray
        EEG signals, (number of channels, number of data points)
        shape (525960, 19, 200) for v1.5.2
    n: integer
        length of positive frequency terms of fourier transform

    Returns:
    --------
    numpy.ndarray
        log amplitude of FFT of signals, (number of channels, number of data points)
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
    Calculate and serialize meta information to pickle, in format:
        {
            'train': list of train edf file names,
            'test': list of test edf file names,
            'mean@train': channel-wise mean of train,
            'std@train': channel-wise std of train,
            'mean@test': channel-wise mean of test,
            'std@test': channel-wise std of test,
            'mean_fft@train': channel-wise mean of train fft,
            'std_fft@train': channel-wise std of train fft
        }
    """
    edf_files_train = []
    edf_files_test = []
    for edf_file in edf_files:
        idx = edf_file.split('.')[0].split('_')[0]
        if idx in train_ids:
            edf_files_train.append(edf_file)
        elif idx in test_ids:
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
    parser.add_argument('--duration', type=int, default=60,
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
                        help='root path for output data.')
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

    print("Reading edf and label file names...")
    edf_files = get_edf_file_names(args.resampled_data_dir)
    print(f"Number of resampled edf files: {len(edf_files)}")

    # process resampleing if no files found in resampled path
    if len(edf_files) == 0:
        print("Resampling...")
        edf_files = resample_all(args.raw_data_dir, args.frequency, args.resampled_data_dir)
        print(f"Number of resampled edf files: {len(edf_files)}")

    print("Getting train/test ids...")
    if args.tusz_version == '1.5.2':
        train_ids, test_ids = get_train_test_ids_v1(args.file_markers_dir, args.train_meta, args.test_meta)
    else:
        train_ids, test_ids = get_train_test_ids_v2(args.raw_data_dir)
    print(f"Number of train ids: {len(train_ids)}")
    print(f"Number of test ids: {len(test_ids)}")

    label_files = get_label_file_names(args.tusz_version, args.raw_data_dir)
    print(f"Number of label files: {len(label_files)}")

    print("Calculating and serializing features and labels in dataframe...")
    nedfs, labels = get_features_and_labels(args.tusz_version, edf_files, label_files, args.frequency, args.duration,
                                            args.montage_type, train_ids, test_ids, feature_dir, label_dir)

    print("Calculating and serializing line-wise features and labels...")
    get_linewise_features_and_labels(edf_files, nedfs, labels, args.frequency, args.duration, train_ids, test_ids,
                                     feature_linewise_dir, label_linewise_dir)

    print("Calculating and serializing meta...")
    get_meta(args.montage_type, args.frequency, train_ids, test_ids, edf_files, nedfs, meta_dir)


if __name__ == '__main__':
    main()
