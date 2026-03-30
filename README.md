# Learning Certified Neural Network Controllers Using Contraction and Interval Analysis

https://github.com/user-attachments/assets/9c4111e4-33cd-45a9-ae3c-e65e50c8deda

The code accompanying the submission titled "Learning Certified Neural Network Controllers Using Contraction and Interval Analysis", by Akash Harapanahalli, Samuel Coogan, and Alexander Davydov. 

## Setup and Installation

### 1. Clone (recursive)

First, clone the repository recursively to also clone a development version of `immrax`.

```bash
git clone --recursive https://github.com/gtfactslab/neural_contraction.git
cd neural_contraction
```

### 2. Create a new `conda` environment (recommended)

We recommend setting up a new conda environment to isolate the dependencies (see [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)). 

```bash
conda create -n neural_contraction python=3.12
conda activate neural_contraction
```

### 3. Install JAX

We used JAX version 0.9.2.

**CPU only:**
```bash
pip install "jax[cpu]==0.9.2"
```

**GPU (CUDA 12):**
```bash
pip install "jax[cuda12]==0.9.2"
```

**GPU (CUDA 13):**
```bash
pip install "jax[cuda13]==0.9.2"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for further details.

### 4. Install dependencies

Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Install the local `immrax` directory

Since we installed the requirements from `requirements.txt`, install the local immrax folder with `--no-deps`.

```bash
pip install --no-deps ./immrax
```

## Reproducing the Quadrotor Example

As discussed in the paper, we consider a 10 state quadrotor model.

### Generate plots from the trained model

To reproduce the figures from the paper, and videos, please run the following.

```bash
python plots.py
```

This saves per nominal trajectory plots and videos under `outputs/`. To tile all four videos into a single side-by-side video:

```bash
./gen_stack.sh
```
This command requires `ffmpeg`, and outputs into `outputs/combined.mp4`.

### Train from scratch

To train the model from scratch, run the following script.

```bash
python training.py
```

Trained weights are saved to `NCM/model.eqx` and `Controller/model.eqx`.
