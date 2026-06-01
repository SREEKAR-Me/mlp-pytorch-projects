# MLP PyTorch Projects

End-to-end implementation of Multilayer Perceptrons (MLPs) built from scratch in PyTorch.
Covers regression, imbalanced binary classification, and time-series classification
across three real-world datasets.

---

## Projects

| Project | Type | Key Challenge | Result |
| California Housing | Regression | Feature scaling, overfitting | RMSE: 0.56, R²: 0.76 |
| Credit Card Fraud | Binary Classification | Severe class imbalance (0.17% fraud) | AUROC: 0.946 |

---

## Model Architecture (shared across all projects)

All three projects use the same core MLP block:

Input → [Linear → BatchNorm1d → ReLU → Dropout] × N → Output

- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Regularization: Dropout + BatchNorm + L2 weight decay
- Device: CUDA (GPU-accelerated training)

---

## Project 1 — California Housing Price Prediction

### Dataset
- Source: sklearn.datasets.fetch_california_housing
- 20,640 samples, 8 features
- Target: median house value (in $100,000s)

### Architecture
- Hidden layers: 256 → 128 → 64
- Output: single neuron, no activation (raw regression output)
- Loss: MSELoss

### Results
- Test RMSE: 0.5635
- Test R²: 0.7598
- Training ran for 100 epochs on CUDA
- Best validation loss: 0.3173

---

## Project 2 — Credit Card Fraud Detection

### Dataset
- Source: Kaggle — ULB Machine Learning Group
- 284,807 transactions, 30 features (V1–V28 PCA-transformed + Amount + Time)
- Severe class imbalance: only 0.17% of transactions are fraudulent

### Architecture
- Hidden layers: 256 → 128 → 64
- Output: single logit (no sigmoid — applied internally by loss)
- Loss: BCEWithLogitsLoss with pos_weight (~577) to handle class imbalance
- Metric: AUROC (accuracy is misleading on imbalanced data)

### Results
- Best validation AUROC: 0.9796
- Test AUROC: 0.9460
- Training ran for 50 epochs on CUDA

### Why AUROC and not accuracy?
A model that predicts every transaction as legitimate achieves 99.83% accuracy
but catches zero fraud cases. AUROC measures how well the model separates fraud
from legitimate transactions across all decision thresholds, making it the only
honest metric for this problem.

---

## Setup

### Requirements
- Python 3.8+
- CUDA-compatible GPU (optional but recommended)

### Installation

1. Clone the repository
   git clone https://github.com/SREEKAR-Me/mlp-pytorch-projects.git
   cd mlp-pytorch-projects

2. Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate        # Mac/Linux
   venv\Scripts\activate           # Windows

3. Install dependencies
   pip install -r requirements.txt

---

## Running the Projects

### California Housing
cd California-Housing-Prediction
python train.py
python evaluate.py

### Credit Card Fraud
cd Credit-Card-Fraud-Prediction
python train.py
python evaluate.py

Note: Download creditcard.csv from Kaggle and place it in
Credit-Card-Fraud-Prediction/data/ before running.
The CSV is excluded from this repo due to GitHub's 100MB file size limit.

---

## Key Concepts Covered

- Custom PyTorch Dataset and DataLoader
- MLP with BatchNorm and Dropout from scratch
- Adam optimizer
- Handling severe class imbalance with weighted loss
- Evaluation with RMSE, R², and AUROC
- Preventing data leakage in train/val/test splits
- GPU-accelerated training with CUDA

---

## Author

Sreekar
github.com/SREEKAR-Me
