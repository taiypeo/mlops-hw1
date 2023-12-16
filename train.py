from dataclasses import dataclass
from pathlib import Path

import hydra
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


cs = ConfigStore.instance()
cs.store(name="mnist_config", node=MnistConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def train(cfg: MnistConfig) -> None:
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
    for epoch in trange(cfg.params.epochs):
        losses = []
        for X, y in dataloader:
            optimizer.zero_grad()

            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch} has finished, current average loss: {torch.mean(loss)}")

    model_path = Path(cfg.paths.save_dir) / "model.pt"
    print(f"Finished! Saving the resulting model to {model_path}")
    Path(cfg.paths.save_dir).mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        torch.save(net, file)


if __name__ == "__main__":
    train()
