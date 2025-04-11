# 📦 Installation

We recommend using Python 3.8+ with pip.

## Requirements

```bash
# Core frameworks
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# ⚠ torch-scatter depends on CUDA and PyTorch version
# Please install manually via: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

torch-geometric==2.4.0
dgl==1.1.2
rdkit==2022.9.5
scikit-learn>=1.2.0
numpy>=1.24.0
pandas>=1.3.0
scipy>=1.7.0
tqdm
matplotlib
```

## Installation

```bash
pip install -r requirements.txt
```

### ⚠ Torch-scatter installation example

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```