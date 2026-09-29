
"""
Credit Card Fraud Detection
Dataset: Kaggle Credit Card Fraud Detection (creditcard.csv)

Run:
    python main.py

Expected dataset location:
    data/creditcard.csv

The program:
1. Loads and inspects the dataset.
2. Analyzes class imbalance.
3. Splits data using stratification.
4. Scales Amount and Time.
5. Applies SMOTE to the training data only.
6. Trains Logistic Regression, Random Forest, and XGBoost.
7. Evaluates using precision, recall, F1-score, ROC-AUC, PR-AUC and confusion matrix.
8. Saves metrics, plots, and trained models.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
OUTPUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUTPUT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download the Kaggle Credit Card Fraud Detection dataset and "
            "place creditcard.csv inside the data folder."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum().sum())
    return df

def analyze_imbalance(df):
    counts = df["Class"].value_counts().sort_index()
    percentages = df["Class"].value_counts(normalize=True).sort_index() * 100

    print("\nClass distribution:")
    print(counts)
    print("\nClass percentage:")
    print(percentages.round(4))

    labels = ["Non-Fraud", "Fraud"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values)
    plt.title("Class Distribution")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(OUTPUT / "class_distribution.png", dpi=160)
    plt.close()

def prepare_data(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Time and Amount are not PCA-transformed in the Kaggle dataset,
    # so scale them for models that benefit from comparable feature ranges.
    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # IMPORTANT: SMOTE is applied only to training data to prevent data leakage.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nBefore SMOTE:", y_train.value_counts().to_dict())
    print("After SMOTE:", y_train_smote.value_counts().to_dict())

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    return X_train_smote, X_test, y_train_smote, y_test

def build_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=None, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        print("\nXGBoost is not installed. Install requirements.txt and rerun.")

    return models

def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, prob)
    pr_auc = average_precision_score(y_test, prob)

    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)
    print(classification_report(y_test, pred, digits=4, zero_division=0))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Fraud", "Fraud"])
    disp.plot(values_format="d")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(OUTPUT / f"{safe_name}_confusion_matrix.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{safe_name}_roc_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(recall_curve, precision_curve, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} - Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{safe_name}_pr_curve.png", dpi=160)
    plt.close()

    return {
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }

def main():
    print("CREDIT CARD FRAUD DETECTION")
    print("-" * 60)

    df = load_data()
    analyze_imbalance(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    results = []
    models = build_models()

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)

    results_df = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
    results_df.to_csv(OUTPUT / "model_comparison.csv", index=False)

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT}")
    print(f"Saved models to : {MODEL_DIR}")

if __name__ == "__main__":
    main()
