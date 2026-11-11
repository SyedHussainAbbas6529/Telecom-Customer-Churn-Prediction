# Telecom Customer Churn Prediction Using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-brightgreen.svg)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Complete%20%26%20Verified-success.svg)]()

---

## Academic Project Information

* **Project Title:** Telecom Customer Churn Prediction Using Machine Learning
* **Project Type:** Supervised Machine Learning — Binary Classification
* **Academic Level:** Undergraduate, Bachelor of Science in Artificial Intelligence (BS AI)
* **Team Members:**
  * **Abdul Rehman** — SAP ID: `70176650`
  * **Syed Hussain Abbass** — SAP ID: `70176788`
* **Target Variable:** `Churn` (Binary: `Yes` = 1, `No` = 0)
* **Primary Environment:** Jupyter Notebook (`.ipynb`)
* **Programming Language:** Python 3

---

## Project Overview

Customer attrition (**churn**) is a critical operational challenge in the telecommunications industry. Because customer acquisition costs (CAC) typically exceed customer retention costs by more than five times, predicting which subscribers are likely to terminate service allows telecom operators to intervene proactively with targeted retention offers.

This semester project implements an end-to-end, reproducible, leak-free machine learning workflow. It covers dataset ingestion, data quality assessment, defensible data cleaning, exploratory data analysis, leak-free feature transformation pipelines, stratified cross-validation, hyperparameter optimization (`GridSearchCV` and `RandomizedSearchCV`), final generalization testing on held-out data, model explainability via permutation importance, pipeline serialization with `joblib`, and customer churn prediction.

---

## Dataset Information

* **Dataset Name:** IBM Telco Customer Churn
* **Filename:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
* **Dimensions:** 7,043 rows $\times$ 21 columns
* **Target Variable:** `Churn`
  * Retained (`No`): 5,174 records (73.46%)
  * Churned (`Yes`): 1,869 records (26.54%)
  * Imbalance Ratio: ~2.77 : 1
* **Feature Categories:**
  1. **Customer Identification:** `customerID`
  2. **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  3. **Account Longevity:** `tenure`
  4. **Telephony Services:** `PhoneService`, `MultipleLines`
  5. **Internet Services:** `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  6. **Contract & Billing:** `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

---

## Project Folder Structure

```
Telecom-Customer-Churn-Prediction/
│
├── datasets/
│   └── 8_telecom_customer_churn_prediction/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Authentic IBM Telco dataset
│
├── notebooks/
│   └── 8_telecom_customer_churn_prediction.ipynb # Fully executed, documented notebook
│
├── models/
│   └── best_churn_model.joblib                    # Serialized final pipeline
│
├── requirements.txt                               # Project dependencies
│
└── README.md                                      # Academic documentation and guide
```

---

## Methodology and Project Phases

The project is structured into 14 sequential phases in compliance with academic machine learning standards:

1. **Phase 1: Project Setup and Introduction** — Formulates the supervised classification problem, sets global random state (`RANDOM_STATE = 42`), and documents academic metadata.
2. **Phase 2: Dataset Loading, Understanding, and Data Quality Assessment** — Audits data dimensions, schema, and detects 11 hidden whitespace strings in `TotalCharges`.
3. **Phase 3: Data Cleaning** — Coerces `TotalCharges` to `float64`, preserves all 7,043 rows, and delegates imputation to training-fold transformers to prevent data leakage.
4. **Phase 4: Exploratory Data Analysis (EDA)** — Visualizes relationships between churn and tenure, monthly charges, contracts, internet services, payment methods, and add-on subscriptions.
5. **Phase 5: Feature Preparation and Train-Test Split** — Drops `customerID`, applies stratified 80/20 train-test partitioning, and constructs scikit-learn `Pipeline` and `ColumnTransformer` modules.
6. **Phase 6: Baseline and Machine Learning Model Training** — Trains `DummyClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, and `RandomForestClassifier`.
7. **Phase 7: Cross-Validation and Model Evaluation** — Evaluates models via 5-fold Stratified Cross-Validation across Accuracy, Precision, Recall, F1-score, and ROC-AUC.
8. **Phase 8: Hyperparameter Optimization** — Tunes Logistic Regression (`GridSearchCV`) and Random Forest (`RandomizedSearchCV`), achieving large recall improvements via balanced class weighting.
9. **Phase 9: Final Model Evaluation** — Evaluates the champion model on the untouched test set ($N=1,409$).
10. **Phase 10: Feature Importance and Model Interpretation** — Quantifies top predictive churn drivers using permutation importance.
11. **Phase 11: Model Saving and Reloading** — Serializes the entire pipeline with `joblib` and validates lossless reloading.
12. **Phase 12: Customer Churn Prediction** — Demonstrates single and batch customer scoring, risk probability generation, and resilience against unseen categories and missing values.
13. **Phase 13: Results and Discussion** — Compares cross-validation and test metrics, synthesizes EDA insights, discusses technical hurdles, and outlines limitations.
14. **Phase 14: Final Project Conclusion** — Reviews academic objectives, work completed, learning outcomes, and future research directions.

---

## Key Experimental Results

### Model Benchmark (5-Fold Stratified Cross-Validation vs. Final Test Set)

| Model Architecture | Evaluation Setting | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | 5-Fold Stratified CV | 73.46% | 0.00% | 0.00% | 0.0000 | 0.5000 |
| **Decision Tree (Default)** | 5-Fold Stratified CV | 72.99% | 49.13% | 49.77% | 0.4941 | 0.6559 |
| **Random Forest (Default)** | 5-Fold Stratified CV | 78.43% | 62.37% | 47.49% | 0.5387 | 0.8177 |
| **Logistic Regression (Default)**| 5-Fold Stratified CV | 80.26% | 65.47% | 54.31% | 0.5930 | 0.8461 |
| **Logistic Regression (Tuned)** | 5-Fold Stratified CV | 74.30% | 52.12% | 79.33% | 0.6289 | 0.8458 |
| **Random Forest (Tuned)** | 5-Fold Stratified CV | 76.13% | 54.14% | 76.12% | **0.6325** | **0.8415** |
| **Random Forest (Champion)** | **Held-Out Test Set (20%)** | **75.59%** | **52.79%** | **75.94%** | **0.6228** | **0.8407** |

### Test Set Confusion Matrix Breakdown ($N = 1,409$)

* **True Negatives (TN):** 781 retained customers correctly identified (75.46% specificity)
* **False Positives (FP):** 254 non-churners flagged for retention outreach (false alarms)
* **False Negatives (FN):** 90 churners missed (only 24.06% missed)
* **True Positives (TP):** **284 churners successfully captured (75.94% sensitivity / recall)**

---

## Installation & Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed:
```bash
python --version
```

### 2. Clone or Download the Project
Navigate to your project directory:
```bash
cd Telecom-Customer-Churn-Prediction
```

### 3. Create and Activate a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install all required libraries from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## How to Run the Notebook

### Launching Jupyter Notebook
You can launch Jupyter from either the project root or the `notebooks/` directory. The notebook automatically locates the dataset using relative paths.

From the project root:
```bash
jupyter notebook notebooks/8_telecom_customer_churn_prediction.ipynb
```
Or simply start the Jupyter server:
```bash
jupyter notebook
```
and open `8_telecom_customer_churn_prediction.ipynb` in the browser interface.

### Running the Complete Pipeline
1. Select **Kernel** $\rightarrow$ **Restart & Run All**.
2. All 75 code cells will execute sequentially in approximately 45–60 seconds.
3. All plots, confusion matrices, ROC curves, feature importance charts, and tables will render in order.

---

## Authors & Acknowledgments

* **Abdul Rehman** (SAP ID: `70176650`) — BS Artificial Intelligence
* **Syed Hussain Abbass** (SAP ID: `70176788`) — BS Artificial Intelligence
* **Dataset Reference:** IBM Telco Customer Churn (Kaggle: `blastchar/telco-customer-churn`)
* **Reference Implementation:** [Telco Customer Churn Prediction Repository](https://github.com/ShefaaSaied/Telco-customer-churn-prediction)
