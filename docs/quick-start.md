In this turorial, we give two examples with UEA & UCR Time Series Classification and Regression Repository.

## Classification

Here we take the `Heartbeat` classification task from [UEA & UCR Time Series Classification Repository](http://timeseriesclassification.com/dataset.php) as an example.

1. Download the dataset
```bash
cd PhysioPro
mkdir data
wget http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip -P data

unzip data/Multivariate2018_ts.zip -d data/Multivariate2018_ts/
rm data/Multivariate2018_ts.zip
```

2. Run `Heartbeat` classification task with `TSRNN` model
```bash
# create the output directory
mkdir -p outputs/Multivariate2018_ts/Heartbeat
# run the train task
python -m physiopro.entry.train docs/configs/rnn_classification.yml
```

The results will be saved to `outputs/Multivariate2018_ts/Heartbeat` directory. 

## Regression

Here we take the `BeijingPM25Quality` dataset from [UEA & UCR Time Series Extrinsic Regression Dataset](http://tseregression.org/) as an example.

1. Download the dataset
```bash
mkdir -p data/Monash_UEA_UCR_Regression_Archive/BeijingPM25Quality
wget https://zenodo.org/record/3902671/files/BeijingPM25Quality_TEST.ts?download=1 -O data/Monash_UEA_UCR_Regression_Archive/BeijingPM25Quality/BeijingPM25Quality_TEST.ts
wget https://zenodo.org/record/3902671/files/BeijingPM25Quality_TRAIN.ts?download=1 -O data/Monash_UEA_UCR_Regression_Archive/BeijingPM25Quality/BeijingPM25Quality_TRAIN.ts
```

2. Run `BeijingPM25Quality` regression task with `TSRNN` model
```bash
# create the output directory
mkdir -p outputs/Monash_UEA_UCR_Regression_Archive/BeijingPM25Quality
# run the train task
python -m physiopro.entry.train docs/configs/rnn_regression.yml
```

The results will be saved to `outputs/Monash_UEA_UCR_Regression_Archive/BeijingPM25Quality` directory.
