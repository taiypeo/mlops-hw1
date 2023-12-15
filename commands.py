from fire import Fire

from mnist_classifier import infer, train


class MnistClassifier:
    """Yet another MNIST classifier"""

    def __init__(self):
        self.train = train
        self.infer = infer


if __name__ == "__main__":
    Fire(MnistClassifier)
