# MAF-GNN

Created by [Zhihong Chen](https://github.com/ZHChen-294), Jiang Wu, Li Pu, Shaoqing Li, Yuhao Zhang, [Dezhong Yao](https://scholar.google.com.hk/citations?user=ClUoWqsAAAAJ&hl=zh-CN&oi=ao)* and Hongmei Yan*

<!-- _The Clinical Hospital of Chengdu Brain Science Institute, Sichuan Institute for Brain Science and Brain-Inspired Intelligence, School of Life Science and
Technology, University of Electronic Science and Technology of China, Chengdu, 610054, Sichuan, China_ -->

This repository contains PyTorch implementation for MAF-GNN.

Paper: Graph Neural Network-based Multi-Atlas Brain Functional Information Fusion for Major Depressive Disorder Diagnosis with rs-fMRI
<!-- Submitted to [**Biomedical Signal Processing and Control**](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control) (In Revising). -->

Code: https://github.com/ZHChen-294/MAF-GNN

<div align="center">
  <img src="https://github.com/ZHChen-294/MAC-GNN/blob/main/Img/MAF-GNN.png">
</div>


# 🧠 Project Dependencies

This project was developed in **Python 3.8+**, and relies on the following main libraries.

---

## 📦 Dependencies Summary

| Category | Packages |
|-----------|-----------|
| **Deep Learning & Graph Neural Networks** | `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data`, `torch_geometric`, `einops` |
| **Scientific Computing & Data Analysis** | `numpy`, `pandas`, `scipy`, `scikit-learn` |
| **Utilities & System** | `os`, `sys`, `time`, `copy`, `random`, `argparse`, `datetime` |
| **Project Modules (Local)** | `utils`, `data_loading`, `Config`, `Model.MAF_GNN` |

---

## ⚙️ Installation

You can install the required dependencies using:

```bash
pip install -r requirements.txt

