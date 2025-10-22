# MAF-GNN

<p align="center">
Created by <a href="https://github.com/ZHChen-294">Zhihong Chen</a>, Jiang Wu, Li Pu, Shaoqing Li, Yuhao Zhang, 
<a href="https://scholar.google.com.hk/citations?user=ClUoWqsAAAAJ&hl=zh-CN&oi=ao">Dezhong Yao*</a> and Hongmei Yan*
</p>

<!-- _The Clinical Hospital of Chengdu Brain Science Institute, Sichuan Institute for Brain Science and Brain-Inspired Intelligence, 
School of Life Science and Technology, University of Electronic Science and Technology of China, Chengdu, 610054, Sichuan, China_ -->

---

This repository contains the **PyTorch implementation** for **MAF-GNN**.

**Paper:**  
*Graph Neural Network-based Multi-Atlas Brain Functional Information Fusion for Major Depressive Disorder Diagnosis with rs-fMRI*  
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

## ⚙️ Installation

Clone the repository and install all dependencies:

```bash
git clone https://github.com/ZHChen-294/MAF-GNN.git
cd MAF-GNN
pip install -r requirements.txt
