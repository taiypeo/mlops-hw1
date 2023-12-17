# mlops-hw1
Yet another MNIST digit classifier

## Usage
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
