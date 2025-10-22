# MAF-GNN

<p align="center">
Created by <a href="https://github.com/ZHChen-294">Zhihong Chen</a>, Jiang Wu, Li Pu, Shaoqing Li, Yuhao Zhang, 
<a href="https://scholar.google.com.hk/citations?user=ClUoWqsAAAAJ&hl=zh-CN&oi=ao">Dezhong Yao*</a> and Hongmei Yan*
</p>

<!-- _The Clinical Hospital of Chengdu Brain Science Institute, Sichuan Institute for Brain Science and Brain-Inspired Intelligence, 
School of Life Science and Technology, University of Electronic Science and Technology of China, Chengdu, 610054, Sichuan, China_ -->

---

This repository contains the **PyTorch implementation** for **MAF-GNN**.

**Paper:** *Graph Neural Network-based Multi-Atlas Brain Functional Information Fusion for Major Depressive Disorder Diagnosis with rs-fMRI*  
<!-- Submitted to [**Biomedical Signal Processing and Control**](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control) (In Revising). -->

**Code:** [https://github.com/ZHChen-294/MAF-GNN](https://github.com/ZHChen-294/MAF-GNN)

<div align="center">
  <img src="https://github.com/ZHChen-294/MAC-GNN/blob/main/Img/MAF-GNN.png" width="70%">
</div>

---

# 🧠 Project Dependencies

This project was developed in **Python 3.8+** and implemented with **PyTorch**.

---

## 📦 Dependencies Summary

| Category | Packages |
|-----------|-----------|
| **Deep Learning / Graph Neural Networks** | `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data`, `torch_geometric`, `einops` |
| **Scientific Computing / Data Analysis** | `numpy`, `pandas`, `scipy`, `scikit-learn` |
| **Utilities / System** | `os`, `sys`, `time`, `copy`, `random`, `argparse`, `datetime` |

---

## 💻 Hardware Requirements

The model requires a **GPU for training acceleration**, and it is recommended to use **NVIDIA RTX 3060 or higher (≥12 GB VRAM)**.

| Component | Minimum | Recommended |
|------------|-----------|-------------|
| **GPU** | NVIDIA RTX 3060 (12 GB VRAM) | RTX 3090 or higher |
| **CPU** | 6 cores | 12 cores or more |
| **RAM** | 16 GB | 32 GB or more |
| **Storage** | 20 GB free disk space | 100 GB (for multi-fold training) |
| **OS** | Ubuntu 20.04+ / Windows 10+ | — |
| **CUDA Toolkit** | 11.8 – 12.1 | Match your PyTorch version |
| **Python** | ≥ 3.8 | 3.8 recommended |

---

## ⚙️ Installation

Clone the repository and install all dependencies:

```bash
git clone https://github.com/ZHChen-294/MAF-GNN.git
cd MAF-GNN
pip install -r requirements.txt
