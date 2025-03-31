# scMIAC

Single-Cell Multi-modality Integration via cell type filtered Anchors using Contrastive learning

<img src="https://github.com/Telogen/scMIAC/blob/main/figures/Fig1.png" width="800">

## Installation

- Create a conda environment

```
conda create -n scMIAC python=3.8
conda activate scMIAC
```

- [Install PyTorch according to your CUDA version](https://pytorch.org/get-started/previous-versions/)

```
# example:
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- Install scMIAC

```
wget https://github.com/Telogen/scMIAC/archive/refs/heads/main.zip
pip install main.zip
```

## Usage

### Data preprocessing

**Notes: Ensure:**

- The **normalized expression matrix of all genes** is in `.raw` attribute.
- The **normalized activity matrix of all genes** is in `.raw` attribute.
- The **predicted cell type labels** are in `.obs['pred']` attribute.

Example: https://github.com/Telogen/scMIAC/blob/main/tutorial/preprocess.ipynb

### scMIAC for diagonal integration

Example: https://github.com/Telogen/scMIAC/blob/main/tutorial/demo_diagonal.ipynb

### scMIAC for vertical integration

Example: https://github.com/Telogen/scMIAC/blob/main/tutorial/demo_vertical.ipynb

### scMIAC for horizontal integration

Refer to: https://github.com/Telogen/scMIAC/blob/main/tutorial/demo_diagonal.ipynb

## Contact

ljtian20@fudan.edu.cn
