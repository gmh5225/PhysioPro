# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path
from shutil import rmtree
import physiopro.entry.train_post_ensemble as train

output_dir = Path("./outputs")


def test_rnn_train():
    train.run_train(
        train.ForecastConfig.fromfile(Path(__file__).parent / "configs" / "rnn_classification_post_ensemble.yml")
    )
    rmtree("outputs")
    # train.run_train(train.ForecastConfig.fromfile(Path(__file__).parent / "configs" / "rnn_regression.yml"))


if __name__ == "__main__":
    test_rnn_train()
