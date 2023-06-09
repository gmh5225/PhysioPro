
This document describes how to preprocess the TUH EEG Seizure Corpus (TUSZ) dataset. The data can be downloaded from 
[Temple University Hospital EEG Seizure Corpus (TUSZ)  v2.0.0](https://isip.piconepress.com/projects/tuh_eeg/index.shtml) 

## Scripts

For the parameter `file-markers-dir`: clone the file markers from `https://github.com/tsy935/eeg-gnn-ssl/tree/main/data/file_markers_ssl` and set it as `<download_path>/eeg-gnn-ssl/data/file_markers_detection`.
This is for train/test splitting, no longer needed for TUSZ v2.0.0.

The montage are defined `scripts/montage.py`.

Run `python scripts/tusz.py -h` to see help messages.

```bash
# v1.5.2
python scripts/tusz.py \
    --tusz-version 1.5.2 \
    --raw-data-dir data/public_dataset_tusz/ \
    --output-dir data/tusz_processed_v1.5.2/ \
    --train-meta trainSet_seq2seq_60s_sz.txt trainSet_seq2seq_60s_nosz.txt \
    --test-meta testSet_seq2seq_60s_sz.txt testSet_seq2seq_60s_nosz.txt

# v2.0.0 not supported yet
```
