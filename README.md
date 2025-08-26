## **MolGraph-xLSTM**  
MolGraph-xLSTM is a graph-based dual-level xLSTM framework designed for molecular property prediction. This model improves molecular feature extraction and enhances prediction accuracy for both classification and regression tasks.

![MolGraph-xLSTM Architecture](mol-xlstm.png)

## **Requirements**  
To ensure reproducibility, all dependencies are listed in `requirements.txt`. Below are the tested installation steps for setting up the environment on **Linux (Ubuntu 22.04)** using **Conda and Python 3.10.0**.

## **Installation**  
Clone the repository and set up the Conda environment:  

git clone https://github.com/syan1992/MolGraph-xLSTM  
cd MolGraph-xLSTM  

conda create -n molgraph-xlstm python=3.10.0 -y  
conda activate molgraph-xlstm  

pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.0+cu121 torchvision torchaudio  
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html  
pip install -r requirements.txt  

## **Data**
The following datasets are used in our experiments:
1. MoleculeNet Dataset

| **Dataset**  | **Samples** | **Task Type** |
|-------------|------------|--------------|
| **BACE**    | 1.5k       | Binary Classification |
| **BBBP**    | 2.0k       | Binary Classification |
| **HIV**     | 41.1k      | Binary Classification |
| **ClinTox** | 1.5k       | Binary Classification |
| **Sider**   | 1.4k       | Binary Classification |
| **Tox21**   | 7.8k       | Binary Classification |
| **ESOL**    | 1.1k       | Regression |
| **Freesolv**| 0.6k       | Regression |
| **Lipo**    | 4.2k       | Regression |
| **Caco2**   | 0.9k       | Regression |

Each dataset was split into **training (80%), validation (10%), and test (10%)** subsets.  
All partitioned datasets are located in the `datasets` folder.

2. TDC Dataset
   
| **Dataset**  | **Samples** | **Task Type** |
|-------------|------------|--------------|
| **HIA**    | 578       | Binary Classification |
| **Pgp**    | 1212       | Binary Classification |
| **Bioavailability** 640    | 41.1k      | Binary Classification |
| **CYP2D6-I** | 13130       | Binary Classification |
| **CYP3A4-I**   | 12328       | Binary Classification |
| **CYP2C9-I**   | 12092       | Binary Classification |
| **hERG**    | 648       | Binary Classification |
| **AMES**| 7255       | Binary Classification |
| **DILI**    | 475       | Binary Classification |
| **Caco2**   | 906       | Regression |
| **PPBR**    | 1797       | Regression |
| **LD50**   | 7385       | Regression |

Each dataset was split into **training (70%), validation (10%), and test (20%)** subsets.  
All partitioned datasets are located in the `datasets` folder.

## **Running Code**
1. Classification  
   python main.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 100 --trial 44 --dataset bace --num_tasks 1 --classification --num_blocks 2
   --slstm 0 --data_dir "datasets" --num_gc_layers 4 --power 4 --num_dim 128 --num_experts 4 --num_heads 4
2. Regression  
   python main.py --lr_decay_epochs 800 --lr_decay_rate 0.5 --learning_rate 0.0002 --batch_size 128 --epochs 200 --trial 41 --dataset freesolv --num_tasks 1 --num_blocks 2
   --slstm 0 --data_dir "datasets" --num_gc_layers 4  --power 4 --num_dim 128 --dropout 0.5 --mlp_layer 1 --num_experts 8 --num_heads 8

## **Hyperparameter Setting**

Moleculenet Dataset Hyperparameter Setting
| | power | dimension | #experts | #heads | #expert layer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BACE | 4 | 256 | 8 | 16 | 2 |
| BBBP | 4 | 256 | 8 | 8 | 2 |
| HIV | 2 | 128 | 4 | 8 | 2 |
| ClinTox | 4 | 128 | 8 | 8 | 1 |
| Sider | 4 | 128 | 8 | 8 | 1 |
| Tox21 | 4 | 128 | 8 | 8 | 2 |
| Freesolv | 4 | 256 | 8 | 8 | 1 |
| ESOL | 4 | 256 | 8 | 8 | 1 |
| Lipo | 4 | 128 | 8 | 8 | 2 |

TDC Dataset Hyperparameter Setting
| | power | dimension | #experts | #heads | #expert layer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| HIA | 4 | 32 | 4 | 4 | 2 |
| Pgp | 4 | 128 | 4 | 4 | 2 |
| Bioav | 4 | 128 | 4 | 4 | 2 |
| CYP2D6-I | 4 | 64 | 4 | 4 | 2 |
| CYP3A4-I | 4 | 128 | 4 | 4 | 2 |
| CYP2C9-I | 4 | 128 | 4 | 4 | 2 |
| hERG | 4 | 32 | 4 | 4 | 2 |
| AMES | 4 | 64 | 4 | 4 | 2 |
| DILI | 4 | 128 | 8 | 8 | 2 |
| Caco2 | 4 | 128 | 4 | 4 | 2 |
| PPBR | 4 | 64 | 4 | 4 | 2 |
| LD50 | 4 | 128 | 4 | 4 | 2 |

## **License**
This project is licensed under the MIT License.

## **Acknowledgement**
Supervised contrastive learning : https://github.com/HobbitLong/SupContrast  
mixture-of-expert: https://github.com/lucidrains/mixture-of-experts/tree/master
