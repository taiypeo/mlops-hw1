import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import trange

from .network import ConvNet

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def train(
    data_dir: str, save_dir: str, batch_size: int, epochs: int, lr: float
) -> None:
    mnist_train = MNIST(root=data_dir, train=True, download=False, transform=ToTensor())
    dataloader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
    net = ConvNet()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    print(
        f"Started training the model. The training will take {epochs} epochs, "
        f"{len(dataloader)} batches each"
    )
    for epoch in trange(epochs):
        losses = []
        for X, y in dataloader:
            optimizer.zero_grad()

            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch} has finished, current average loss: {torch.mean(loss)}")

    model_path = Path(save_dir) / "model.pt"
    print(f"Finished! Saving the resulting model to {model_path}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        torch.save(net, file)


def infer(
    data_dir: str, model_dir: str, predictions_path: str, batch_size: int
) -> None:
    mnist_test = MNIST(root=data_dir, train=False, download=False, transform=ToTensor())
    dataloader = DataLoader(dataset=mnist_test, batch_size=batch_size)
    with open(Path(model_dir) / "model.pt", "rb") as file:
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
    print(f"Saving predictions to {predictions_path}")
    with open(predictions_path, "w") as file:
        json.dump(predictions, file)
