

### `README.md`

```markdown
# M2 Variational Autoencoder (VAE) Training

This repository contains the code to train M2 VAEs with either the standard M2 loss or the optimized ELBO loss(Reserch extention). The models support training on MNIST and CIFAR-10 datasets.

---

## Prerequisites

1. **Python**: Ensure you have Python 3.8 or later installed.
2. **Conda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

---

## Setup Instructions

### Step 1: T0 create and Activate the Conda Environment

```bash
# Create a new environment named "vae-env"
conda create -n vae-env python=3.8 -y

# Activate the environment
conda activate vae-env
```

---

### Step 2: Install Dependencies

#### Install PyTorch
Install PyTorch with CUDA 11.8 support:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you do not have a CUDA-capable GPU, install the CPU version of PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Install Other Required Libraries
```bash
pip install matplotlib numpy
```

---

### Step 3: Dataset Preparation

The MNIST and CIFAR-10 datasets will be automatically downloaded and saved in the `../data` directory when the script is run. No manual setup is required for datasets.

---

## How to Run the Code

### Training on MNIST

#### Standard M2 Loss:
```bash
python ../src/main.py --dataset MNIST
```

#### Optimized ELBO Loss:
```bash
python ../src/main.py --optimized_elbo --dataset MNIST
```

---

### Training on CIFAR-10

#### Standard M2 Loss:
```bash
python ../src/main.py --dataset CIFAR10
```

#### Optimized ELBO Loss:
```bash
python ../src/main.py --optimized_elbo --dataset CIFAR10
```

---

