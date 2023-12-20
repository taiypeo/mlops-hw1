import subprocess
from dataclasses import dataclass
from pathlib import Path

import hydra
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import trange

from mnist_classifier import setup_random_seeds
from mnist_classifier.network import ConvNet


# https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


@dataclass
class PathConfig:
    data_dir: str
    save_dir: str


@dataclass
class TrainParams:
    batch_size: int
    epochs: int
    lr: float
    n_conv_channels: int
    n_linear_out_channels: int
    learn_bias: bool


@dataclass
class MnistConfig:
    paths: PathConfig
    params: TrainParams
    mlflow_uri: str


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def train(cfg: MnistConfig) -> None:
    mlflow.set_tracking_uri(uri=cfg.mlflow_uri)
    mlflow.set_experiment("Train MNIST classifier")

    setup_random_seeds()

    mnist_train = MNIST(
        root=cfg.paths.data_dir, train=True, download=False, transform=ToTensor()
    )
    dataloader = DataLoader(
        dataset=mnist_train, batch_size=cfg.params.batch_size, shuffle=True
    )
    net = ConvNet(
        n_conv_channels=cfg.params.n_conv_channels,
        n_linear_out_channels=cfg.params.n_linear_out_channels,
        learn_bias=cfg.params.learn_bias,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.params.lr)

    print(
        "Started training the model. "
        f"The training will take {cfg.params.epochs} epochs, "
        f"{len(dataloader)} batches each"
    )

    with mlflow.start_run():
        params = {
            "commit_hash": get_git_hash(),
            **dict(cfg.paths),
            **dict(cfg.params),
        }
        mlflow.log_params(params)
        for epoch in trange(cfg.params.epochs):
            losses = []
            correct = 0
            total = 0
            for X, y in dataloader:
                optimizer.zero_grad()

                y_hat = net(X)
                loss = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                batch_predictions = y_hat.argmax(dim=1)
                correct += (batch_predictions == y).sum().item()
                total += y.size(0)

            mlflow.log_metric("Epoch", epoch)

            mean_loss = sum(losses) / len(losses)
            mlflow.log_metric("Mean loss", mean_loss)

            running_accuracy = correct / total
            mlflow.log_metric("Running accuracy", running_accuracy)

            print(
                f"Epoch {epoch} has finished, "
                f"current average loss: {mean_loss}, "
                f"running accuracy: {running_accuracy}"
            )

    model_path = Path(cfg.paths.save_dir) / "model.pt"
    print(f"Finished! Saving the resulting model to {model_path}")
    Path(cfg.paths.save_dir).mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        torch.save(net, file)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="mnist_config", node=MnistConfig)

    train()
