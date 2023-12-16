# mlops-hw1
Yet another MNIST digit classifier

## Usage
### Loading the dataset
No credentials are needed to pull the MNIST dataset from the S3 bucket.
Credentials are only needed to write to it.
```bash
dvc pull
```

### Training
1. Change `conf/train_config.yaml`
2. Run `python train.py`

### Inference on the test set
1. Change `conf/infer_config.yaml`
2. Run `python infer.py`
