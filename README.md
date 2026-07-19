# 🏦 LendingClub Loan Default Predictor

An industry-grade end-to-end Machine Learning pipeline to predict loan default risks for LendingClub using tabular applicant data. This project showcases data cleaning, feature engineering, class imbalance handling, and model training using **XGBoost**, **Random Forest**, and an **Artificial Neural Network (ANN)** built with TensorFlow/Keras.

---

## 🎯 Business Context & Objectives

LendingClub is the largest online peer-to-peer (P2P) loan marketplace. Lending loans to "risky" applicants is the largest source of financial loss (credit loss). Identifying indicators of loan defaults allows LendingClub to minimize credit loss by:
* Denying loans to high-risk applicants.
* Adjusting interest rates to offset risk.
* Reducing approved loan amounts.

The goal of this project is to build a classification pipeline that analyzes past applicant profiles to predict whether a customer will "default" (charged-off) or "repay" (fully paid) their loan.

---

## 🛠️ Tech Stack & Libraries

* **Core Language:** Python 3
* **Data Processing & EDA:** Pandas, NumPy, SciPy, Seaborn, Matplotlib, hvPlot
* **Machine Learning Models:** Scikit-Learn, XGBoost
* **Deep Learning Framework:** TensorFlow & Keras
* **Configuration Management:** PyYAML

---

## 📁 Repository Structure

```text
Lending_Club_Loan_Default_predictor/
│
├── config/
│   └── config.yaml             # Hyperparameters, paths, and preprocessing parameters
│
├── data/
│   ├── raw/                    # Raw Kaggle LendingClub dataset (git-ignored)
│   └── processed/              # Processed train/test splits (git-ignored)
│
├── models/                     # Saved model artifacts (git-ignored)
│   ├── ann_model.keras         # Saved TensorFlow Keras Model
│   ├── xgb_model.json          # Saved XGBoost Classifier
│   ├── rf_model.pkl            # Serialized Random Forest Classifier
│   └── scaler.pkl              # Fitted MinMaxScaler
│
├── notebooks/
│   └── lending-club-loan-defaulters-prediction.ipynb  # Interactive EDA & initial experiments
│
├── plots/                      # Generated evaluation charts (git-ignored)
│   ├── ann_training_history.png
│   ├── confusion_matrix_ann.png
│   ├── confusion_matrix_xgboost.png
│   ├── confusion_matrix_random_forest.png
│   └── roc_curve_comparison.png
│
├── scripts/
│   └── run_pipeline.py         # Entrypoint script to run the end-to-end ML pipeline
│
├── src/                        # Modular library source code
│   ├── __init__.py
│   ├── data_preprocessing.py   # Cleaning, imputations, encoding, splits, and outlier filters
│   ├── model.py                # Model builders for ANN, XGBoost, and Random Forest
│   ├── evaluate.py             # Performance reporting and plotting utilities
│   └── train.py                # Pipeline orchestration and serialization logic
│
├── .gitignore                  # Git exclude configurations
├── requirements.txt            # Software dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation

Clone this repository and install dependencies in a virtual environment:

```bash
# Clone the repository
git clone https://github.com/gauravkasaudhan06/Lending-Club-Loan-Default-predictor.git
cd Lending-Club-Loan-Default-predictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Dataset Setup

1. Download the LendingClub dataset (`lending_club_loan_two.csv`) from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
2. Create the data directories and place the file inside:
   ```bash
   mkdir -p data/raw
   mv lending_club_loan_two.csv data/raw/
   ```

---

## 💻 Running the ML Pipeline

You can run the end-to-end training and evaluation pipeline from the command line:

```bash
python scripts/run_pipeline.py
```

### Options
To use a custom configuration file:
```bash
python scripts/run_pipeline.py --config config/custom_config.yaml
```

When you execute the script, it will:
1. Load config parameters.
2. Load and clean the raw data (imputing missing mortgage accounts, extracting features, encoding categorical columns).
3. Split the data and apply outlier filters **only** to the training split.
4. Scale columns using a `MinMaxScaler`.
5. Train all three models (TensorFlow ANN, XGBoost, Random Forest).
6. Evaluate results and save plots (Loss curves, Confusion Matrices, ROC-AUC comparison curves) to `plots/`.
7. Serialize and save the models to `models/`.

---

## 📊 Evaluation Results

Once run, the models yield performance metrics (ROC-AUC scores) comparing validation performances:

| Model | Train ROC-AUC | Test ROC-AUC | Description |
|---|---|---|---|
| **Artificial Neural Network (ANN)** | ~0.90 | ~0.89 | Dense layers with Batch Normalization & Dropout regularizations. |
| **XGBoost Classifier** | ~0.92 | ~0.90 | Optimized Gradient Boosting Trees. |
| **Random Forest Classifier** | ~0.99 | ~0.87 | Ensemble bagger of Decision Trees. |

*Visual representations of ROC Curves and Confusion Matrices will be automatically saved under the `plots/` directory.*

---

## 🛡️ License

This project is open-source and licensed under the [MIT License](LICENSE).
