import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE

SEED            = 42
DATASET_PATH    = "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv" # https://www.kaggle.com/datasets/blastchar/telco-customer-churn
OUTPUT_PATH     = "outputs/task2_churn_evaluation_results.png"

# Data Preprocessing
df = pd.read_csv(DATASET_PATH)

df = df.drop(columns=['customerID'])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

print(y.value_counts(normalize=True).rename({0: 'Subscription (0)', 1: 'Churn (1)'}) * 100)
print("-" * 50)

# Train-Test Split (Stratified Split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Model Baseline (Without Resampling)
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_baseline.fit(X_train, y_train)

y_pred_base = rf_baseline.predict(X_test)
y_prob_base = rf_baseline.predict_proba(X_test)[:, 1]

# Resampling with SMOTE
smote = SMOTE(random_state=SEED)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Training sample before SMOTE: {y_train.value_counts().to_dict()}")
print(f"Training sample after SMOTE : {y_train_resampled.value_counts().to_dict()}")
print("-" * 50)

rf_smote = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_smote.fit(X_train_resampled, y_train_resampled)

y_pred_smote = rf_smote.predict(X_test)
y_prob_smote = rf_smote.predict_proba(X_test)[:, 1]

# Output
def evaluate_performance(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    }

metrics_base = evaluate_performance(y_test, y_pred_base, y_prob_base, "Baseline (Imbalanced)")
metrics_smote = evaluate_performance(y_test, y_pred_smote, y_prob_smote, "SMOTE (Balanced)")

results_df = pd.DataFrame([metrics_base, metrics_smote])

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cm_base = confusion_matrix(y_test, y_pred_base)
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title("Confusion Matrix - Baseline")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

cm_smote = confusion_matrix(y_test, y_pred_smote)
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title("Confusion Matrix - SMOTE")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_prob_smote)

axes[2].plot(fpr_base, tpr_base, label=f"Baseline (AUC = {metrics_base['ROC-AUC']:.2f})", color='blue')
axes[2].plot(fpr_smote, tpr_smote, label=f"SMOTE (AUC = {metrics_smote['ROC-AUC']:.2f})", color='green')
axes[2].plot([0, 1], [0, 1], 'k--', label="Random Guess")
axes[2].set_title("ROC Curve Comparison")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend()

plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"Visualization saved in: {OUTPUT_PATH}\n")

plt.show()

print(results_df.to_string(index=False))
