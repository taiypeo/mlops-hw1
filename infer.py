import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from mnist_classifier import setup_random_seeds


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


cs = ConfigStore.instance()
cs.store(name="mnist_config", node=MnistConfig)


@hydra.main(version_base=None, config_path="conf", config_name="infer_config")
def infer(cfg: InferenceParams) -> None:
    setup_random_seeds()

    mnist_test = MNIST(
        root=cfg.paths.data_dir, train=False, download=False, transform=ToTensor()
    )
    dataloader = DataLoader(dataset=mnist_test, batch_size=cfg.params.batch_size)
    with open(Path(cfg.paths.model_dir) / "model.pt", "rb") as file:
        net = torch.load(file)
    net.eval()

    print(f"Performing inference on the test data, {len(dataloader)} batches")
    predictions = []
    correct = 0
    total = 0
    with torch.inference_mode():
        for X, y in dataloader:
            y_hat = net(X)

            batch_predictions = y_hat.argmax(dim=1)
            predictions.extend(batch_predictions.tolist())

            correct += (batch_predictions == y).sum().item()
            total += y.size(0)

    print(f"Model's accuracy on the test set: {correct / total}")
    print(f"Saving predictions to {cfg.paths.predictions_path}")
    with open(cfg.paths.predictions_path, "w") as file:
        json.dump(predictions, file)


if __name__ == "__main__":
    infer()
