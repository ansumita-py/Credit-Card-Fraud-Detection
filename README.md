# Credit Card Fraud Detection

A complete data science / machine learning project for detecting fraudulent credit-card transactions using classification and imbalance-handling techniques.

## Objectives
- Analyze fraud vs. non-fraud class imbalance.
- Apply SMOTE to the training data.
- Train Logistic Regression, Random Forest and XGBoost.
- Compare models using Precision, Recall, F1-score, ROC-AUC, PR-AUC and confusion matrices.

## Dataset
Use Kaggle's **Credit Card Fraud Detection** dataset. Download `creditcard.csv` and put it here:

```text
Credit_Card_Fraud_Detection_Project/
└── data/
    └── creditcard.csv
```

The dataset contains a `Class` column:
- `0` = legitimate transaction
- `1` = fraudulent transaction

## Project Structure

```text
Credit_Card_Fraud_Detection_Project/
├── data/
│   └── creditcard.csv              # Add Kaggle dataset here
├── models/                         # Generated after running
├── outputs/                        # Generated charts and metrics
├── main.py
├── requirements.txt
├── README.md
└── project_report.pdf
```

## Installation

Open the project folder in VS Code and run:

```bash
pip install -r requirements.txt
```

Then place `creditcard.csv` in the `data` folder and run:

```bash
python main.py
```

## Important methodology

SMOTE is applied **only to the training set**, not the test set. This avoids test-data leakage and keeps evaluation closer to a real-world setting.

## Expected outputs

After execution, the `outputs` folder will contain:
- `class_distribution.png`
- confusion matrices
- ROC curves
- precision-recall curves
- `model_comparison.csv`

The `models` folder will contain saved `.joblib` models and the scaler.

## Resume / LinkedIn description

**Credit Card Fraud Detection | Python, Pandas, Scikit-learn, SMOTE, Random Forest, XGBoost**

Built a machine learning system to detect fraudulent credit-card transactions by analyzing severe class imbalance, applying SMOTE to the training data, training Logistic Regression, Random Forest and XGBoost classifiers, and evaluating performance using precision, recall, F1-score, ROC-AUC, PR-AUC and confusion matrices.

## Skills demonstrated
Python, Pandas, NumPy, Data Preprocessing, Exploratory Data Analysis, Imbalanced Data Handling, SMOTE, Classification, Logistic Regression, Random Forest, XGBoost, Model Evaluation, Data Visualization, Scikit-learn.
