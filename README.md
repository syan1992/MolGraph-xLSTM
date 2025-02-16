# **MolGraph-xLSTM**  
MolGraph-xLSTM is a graph-based dual-level xLSTM framework designed for molecular property prediction. This model improves molecular feature extraction and enhances prediction accuracy for both classification and regression tasks.

![MolGraph-xLSTM Architecture](mol-xlstm.png)

## **Requirements**  
To ensure reproducibility, all dependencies are listed in `requirements.txt`. Below are the tested installation steps for setting up the environment on **Linux (Ubuntu 22.04)** using **Conda and Python 3.10.0**.
---

## **Installation**  
Clone the repository and set up the Conda environment:  

```bash
git clone https://github.com/syan1992/MolGraph-xLSTM
cd MolGraph-xLSTM

conda create -n molgraph-xlstm python=3.10.0 -y
conda activate molgraph-xlstm

pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.0+cu121 torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -r requirements.txt

## **Data**

