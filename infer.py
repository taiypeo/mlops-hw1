import subprocess
from dataclasses import dataclass
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from mnist_classifier import setup_random_seeds


# https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


@dataclass
class PathConfig:
    data_dir: str
    model_dir: str
    predictions_path: str


@dataclass
class InferenceParams:
    batch_size: int


@dataclass
class MnistConfig:
    paths: PathConfig
    params: InferenceParams
    mlflow_uri: str


@hydra.main(version_base=None, config_path="conf", config_name="infer_config")
def infer(cfg: InferenceParams) -> None:
    mlflow.set_tracking_uri(uri=cfg.mlflow_uri)
    mlflow.set_experiment("Infer MNIST classifier")

    setup_random_seeds()

    mnist_test = MNIST(
        root=cfg.paths.data_dir, train=False, download=False, transform=ToTensor()
    )
    dataloader = DataLoader(dataset=mnist_test, batch_size=cfg.params.batch_size)
    with open(Path(cfg.paths.model_dir) / "model.pt", "rb") as file:
        net = torch.load(file)
    net.eval()

    loss_fn = nn.CrossEntropyLoss()

    print(f"Performing inference on the test data, {len(dataloader)} batches")
    with mlflow.start_run():
        params = {
            "commit_hash": get_git_hash(),
            **dict(cfg.paths),
            **dict(cfg.params),
        }
        mlflow.log_params(params)

        losses = []
        predictions = []
        correct = 0
        total = 0
        with torch.inference_mode():
            for X, y in dataloader:
                y_hat = net(X)
                loss = loss_fn(y_hat, y)
                losses.append(loss.item())

                batch_predictions = y_hat.argmax(dim=1)
                predictions.extend(batch_predictions.tolist())

                correct += (batch_predictions == y).sum().item()
                total += y.size(0)

            test_loss = sum(losses) / len(losses)
            mlflow.log_metric("Test loss", test_loss)

            test_accuracy = correct / total
            mlflow.log_metric("Test accuracy", test_accuracy)

    print(f"Model's accuracy on the test set: {test_accuracy}, " f"loss: {test_loss}")

    print(f"Saving predictions to {cfg.paths.predictions_path}")
    df = pd.Series(predictions).rename("predictions").reset_index()
    df.to_csv(cfg.paths.predictions_path, index=False)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="mnist_config", node=MnistConfig)

    infer()
