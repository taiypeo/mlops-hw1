# mlops-hw1
Yet another MNIST digit classifier.

[MNIST](https://en.wikipedia.org/wiki/MNIST_database) is an image dataset that
contains 28x28 images of hand-written digits from 0 to 9.

## Important files
- `train.py` contains the training code for a simple CNN for the MNIST digit classification task
- `infer.py` contains the inference code for the test set of MNIST
- `mnist_classifier/` contains the source code for the CNN model and other PyTorch-related code
- `conf/` contains the Hydra configuration YAML files for the training and inference stages
- `images/MNIST/raw` contains the DVC files for the dataset
- `results` is created after either `train.py` or `infer.py` is finished running
and contains the trained model and the test set predictions

## Usage
### Setting up the project
1. Install [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
2. Run
```bash
python -m venv .venv
source .venv/bin/activate
poetry install
```

### Setting up local pre-commit hooks
1. Run `pre-commit install`
2. Verify with `pre-commit run -a`

### Setting up mlflow
1. Run `mlflow ui --host 0.0.0.0 --port 8080`
2. If you changed one of the parameters in the above command,
don't forget to update `conf/train_config.yaml` and `conf/infer_config.yaml`

### Loading the dataset
No credentials are needed to pull the MNIST dataset from the S3 bucket.
Credentials are only needed to write to it.

Run `dvc pull`.

### Training
1. Change `conf/train_config.yaml`
2. Run `python train.py`

### Inference on the test set
1. Change `conf/infer_config.yaml`
2. Run `python infer.py`
