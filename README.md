# Multisyn: Accurate prediction of synergistic drug combination using a multi-source information fusion framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15194129.svg)](https://doi.org/10.5281/zenodo.15194129)

This repository contains the official implementation of our paper:  
**Multisyn** integrates pharmacophore structure, protein-protein interaction (PPI) networks, and cell line omics to predict synergistic anti-cancer drug combinations.

![Multisyn Architecture](Multisyn.png)

You can find full documentation here:  [https://HuazeLoong.github.io/MultiSyn/](https://HuazeLoong.github.io/MultiSyn/)

## 1. Introduction

Multisyn represents molecules as heterogeneous molecular graphs and predicts drug combination synergy using graph neural networks.  
It provides substructure-level attention and integrates multi-source data, including PPI and cell lines omics profiles.

**Paper Link**: *Coming soon...*

## 1.1 Features

- Drug heterogeneous molecular graph construction based on BRICS fragments  
-  Dual-view cell line integration: expression + PPI fusion features  
- Multi-modal attention-based GNN architecture  

## 1.2 File Structure

```text
multisyn/             ← Project root directory
├── setup.py          ← Packaging and installation configuration
├── requirements.txt  ← Dependency management
├── README.md         ← Project description
└── src/
    └── multisyn/         ← Python package (contains all core source code)
        ├── __init__.py
        ├── model.py
        ├── train.py
        ├── utils.py
        ├── dataset.py
        ├── const.py
        └── prepare_data.py
```

## 1.3 Citation
If you find this repository helpful, please cite our work:

```bibtex

```

# 2. Usage
## 2.1 Requirements
We recommend the following Python environment:
```bash
# ---- Core Deep Learning Framework ----
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# ⚠ torch-scatter must match your PyTorch and CUDA version.
# Manual installation is recommended (see notes below).

# ---- GNN Packages ----
torch-geometric==2.4.0
dgl==1.1.2  # or dgl==1.1.2+cu118 depending on your CUDA version

# ---- Chemistry Toolkit ----
rdkit==2022.9.5  # from conda or RDKit wheels

# ---- ML + Data Processing ----
scikit-learn>=1.2.0
numpy>=1.24.0
pandas>=1.3.0
scipy>=1.7.0

# ---- Optional Utilities ----
tqdm
matplotlib
```

Install core dependencies using:

```bash
pip install -r requirements.txt
```

**Notes on Specific Dependencies**

⚠ torch-scatter
torch-scatter requires a PyTorch- and CUDA-matching build. Use the following command to install a compatible version:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```
You can find more options at: [PyG Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

⚠ rdkit
rdkit is not available on PyPI; it is recommended to install via conda:
```bash
conda install -c rdkit rdkit==2022.9.5
```

## 2.2 Preprocessing
To preprocess the drug combination dataset:

```bash
python prepare_data.py
```

Processed files will be saved to `multisyn\datas\processed`.

## 2.3 Train the Model
To train the Multisyn model:
```bash
python train.py
```
Results will be saved to the `multisyn\datas\results` directory.
