from fire import Fire

from mnist_classifier import train


class MnistClassifier:
    """Yet another MNIST classifier"""

    def train(self, epochs: int) -> None:
        train()

    def infer(self, model: str) -> None:
        pass


if __name__ == "__main__":
    Fire(MnistClassifier)
