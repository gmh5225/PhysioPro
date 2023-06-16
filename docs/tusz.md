
This document describes how to process the TUH EEG Seizure Corpus (TUSZ) dataset.
The data can be downloaded from [Temple University Hospital EEG Seizure Corpus (TUSZ)  v2.0.0](https://isip.piconepress.com/projects/tuh_eeg/index.shtml).
We use term-based bi-class annotations in this task.

## Scripts

1. Data Preprocessing
For the parameter `file-markers-dir`: clone the file markers from `https://github.com/tsy935/eeg-gnn-ssl/tree/main/data/file_markers_ssl` and set it as `<download_path>/eeg-gnn-ssl/data/file_markers_detection`.
This is for train/test splitting, no longer needed for TUSZ v2.0.0.

The montage are defined `scripts/montage.py`.

Run `python scripts/tusz.py -h` to see help messages.

```bash
# v1.5.2
python scripts/tusz.py \
    --tusz-version 1.5.2 \
    --duration 60 \
    --frequency 200 \
    --raw-data-dir data/public_dataset_tusz/ \
    --resampled-data-dir data/tusz_resampled_v1.5.2/  \
    --output-dir data/tusz_processed_v1.5.2/ \
    --train-meta trainSet_seq2seq_60s_sz.txt trainSet_seq2seq_60s_nosz.txt \
    --test-meta testSet_seq2seq_60s_sz.txt testSet_seq2seq_60s_nosz.txt

# v2.0.0
# original frequecies: [250, 256, 400]
python scripts/tusz.py \
    --tusz-version 2.0.0 \
    --duration 30 \
    --frequency 250 \
    --raw-data-dir data/tuh_eeg_seizure/ \
    --resampled-data-dir data/tusz_resampled_v2.0.0/ \
    --output-dir data/tusz_processed_v2.0.0/
```

2. Run TUSZ with Transformer
```bash
# v1.5.2
# create the output directory
mkdir -p outputs/tusz_v1.5.2
# run the train task
python -m physiopro.entry.train docs/configs/transformer_tusz_v1.5.2.yml

# v2.0.0
# create the output directory
mkdir -p outputs/tusz_v2.0.0
# run the train task
python -m physiopro.entry.train docs/configs/transformer_tusz_v2.0.0.yml
```
