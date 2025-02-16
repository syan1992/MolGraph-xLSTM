# **MolGraph-xLSTM**  

MolGraph-xLSTM is a **graph-based dual-level xLSTM framework** designed for **molecular property prediction**. It effectively captures **long-range molecular dependencies** by integrating **Graph Neural Networks (GNNs), xLSTM, and Multi-Head Mixture-of-Experts (MHMoE)**. This model improves molecular feature extraction and enhances prediction accuracy for **both classification and regression tasks**.

![MolGraph-xLSTM Architecture](https://github.com/syan1992/MolGraph-xLSTM/architecture.png)

## **Key Features**
- **Dual-level molecular graph representation:** Incorporates both **atom-level** and **motif-level** graph structures for comprehensive molecular representation.
- **Enhanced long-range information modeling:** Uses **xLSTM** to address GNN limitations in capturing long-range dependencies.
- **Multi-Head Mixture-of-Experts (MHMoE):** Dynamically assigns input features to multiple expert networks for robust feature extraction.
- **Scalable training pipeline:** Supports GPU acceleration and efficient batch processing.

---

## **Requirements**  
To ensure reproducibility, all dependencies are listed in `requirements.txt`. Below are the tested installation steps for setting up the environment on **Linux (Ubuntu 20.04+)** using **Conda and Python 3.10.0**.

### **Dependencies**
- **Python** >= 3.10.0  
- **PyTorch** >= 1.7.1  
- **Torch-Geometric** >= 2.5.3  
- **RDKit** >= 2022.09.05  
- **NumPy**, **pandas**, **scikit-learn**  
- **DeepChem** (for molecular preprocessing)  

---

## **Installation**  

Clone the repository and set up the Conda environment:  

```bash
git clone https://github.com/syan1992/MolGraph-xLSTM
cd MolGraph-xLSTM

conda create -n molgraph-xlstm python=3.10.0 -y
conda activate molgraph-xlstm

pip install -r requirements.txt
