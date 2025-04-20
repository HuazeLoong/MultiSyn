# Integrating Protein-Protein Interaction Network with Omics Data to Predict Anticancer Synergistic Drug Combinations

This project provides a lightweight implementation of the **PRODeepSyn** model to embed cell lines using protein–protein interaction (PPI) networks and omics targets (e.g., gene expression and mutation). It enables generation of cell line representations for downstream drug synergy prediction tasks.

---

## File Description

| File         | Description                        |
|--------------|------------------------------------|
| `const.py`   | Data path configuration            |
| `dataset.py` | Data loading class definitions     |
| `model.py`   | Model architecture (GAT, Cell2Vec) |
| `train.py`   | Train cell line embedding model    |
| `gen_feat.py`| Generate normalized features       |
| `utils.py`   | Model saving and utility functions |

---
# Run Instructions
## Step 1: 
```python
python train.py
```
Train cell embeddings (both GE and MUT by default)
## Step 2: 
```python
python gen_feat.py mdl_ge_128x384_sample mdl_mut_128x384_sample
```
Generate normalized cell features from saved embeddings
## Output: 
`data/cell_feat.npy`
# 📄 Citation
Xiaowen Wang, Hongming Zhu, Yizhi Jiang, Yulong Li, Chen Tang, Xiaohan Chen, Yunjie Li, Qi Liu, Qin Liu
PRODeepSyn: predicting anticancer synergistic drug combinations by embedding cell lines with protein–protein interaction network
Briefings in Bioinformatics, Volume 23, Issue 2, March 2022, bbab587
https://doi.org/10.1093/bib/bbab587
