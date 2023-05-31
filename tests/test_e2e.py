# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path
from shutil import rmtree

from physiopro.entry import train, pretrain_finetune

output_dir = Path("./outputs")


def test_rnn_train():
    train.run_train(train.ForecastConfig.fromfile(Path(__file__).parent / "configs" / "rnn_classification.yml"))
    rmtree("outputs")
    train.run_train(train.ForecastConfig.fromfile(Path(__file__).parent / "configs" / "rnn_regression.yml"))
    train.run_train(train.ForecastConfig.fromfile(Path(__file__).parent / "configs" / "rnn_regression_resume.yml"))
    rmtree("outputs")


def test_rnn_pretrain():
    pretrain_finetune.run_pretrain_finetune(
        pretrain_finetune.PretrainConfig.fromfile(Path(__file__).parent / "configs" / "rnn_classification_cpc.yml")
    )
    rmtree("outputs")
    pretrain_finetune.run_pretrain_finetune(
        pretrain_finetune.PretrainConfig.fromfile(Path(__file__).parent / "configs" / "rnn_regression_cpc.yml")
    )
    rmtree("outputs")


if __name__ == "__main__":
    test_rnn_train()
