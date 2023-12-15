from fire import Fire

from catdog_classifier import train


class CatDogClassifier:
    """Yet another cat/dog classifier"""

    def train(self, epochs: int) -> None:
        train()

    def infer(self, model: str) -> None:
        pass


if __name__ == "__main__":
    Fire(CatDogClassifier)
