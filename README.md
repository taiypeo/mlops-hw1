# mlops-hw1
Yet another MNIST digit classifier

# Usage
## Training
```bash
python commands.py train --data-dir images/ --save-dir result/ --batch-size 1000 --epochs 10 --lr 0.001
```

## Inference on the test set
```bash
python commands.py infer --data-dir images/ --model-dir result/ --predictions-path result/preds.txt --batch-size 1000
```
