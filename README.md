# chest-xray-classifier

Before using PyTorch's `nn.Conv2d` for a medical imaging project, I wanted to understand what it actually does. So I wrote the pieces of a CNN by hand in NumPy, then built the same kind of classifier in PyTorch and trained it on chest X-rays (normal vs pneumonia).

This repo is that learning project. The more complete follow-up, with benchmarks against pretrained models and Grad-CAM, is [pneumonia-xai](https://github.com/oladimeji-balogun/pneumonia-xai).

## What's in here

**`src/models/numpy_net.py`**: a CNN in plain NumPy, forward and backward:

- `conv2D`, `max_pool2d`, `ReLU`, `softmax`, `flatten`
- cross-entropy loss
- backprop for the fully connected layer, ReLU and max pooling
- a `TinyCNN` class that puts them together with a `train_step`

**`src/models/cnn.py`**: the PyTorch versions. `TinyCNN` mirrors the NumPy one, and `ChestXRayCNN` is a larger model with three conv blocks, batch norm and dropout.

**`src/training/`** and **`src/evaluation/`**: a training loop with early stopping and class-weighted loss, and metrics including evaluation at a custom decision threshold.

**`tests/`**: pytest tests for the NumPy layers, the CNN and the trainer.

## Things I learned along the way

- The dataset is about 3:1 pneumonia to normal. Plain accuracy looked good early on and was misleading. Class-weighted loss, a weighted random sampler and a stratified validation split made the numbers more honest.
- The dataset's own validation folder only has 16 images, so I split validation off the training set instead.
- Lowering the decision threshold trades specificity for recall, which matters when missing pneumonia is worse than a false alarm.

## Running it

```bash
poetry install
poetry run pytest
```

Training expects the [Kaggle chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) at `data/raw/chest_xray/`. Then run `python main.py`.
