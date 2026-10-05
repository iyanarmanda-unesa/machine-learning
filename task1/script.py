import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

SEED = 42
DATASET_PATH = "datasets/healthcare-dataset-stroke-data.csv"
OUTPUT_DIR = "outputs"
OUTPUT_FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "task1_feature_importance.png")
OUTPUT_MATRIKS_PATH = os.path.join(OUTPUT_DIR, "task1_confusion_matrix.png")

def preprocess_and_engineer(data: pd.DataFrame) -> pd.DataFrame:
    df_out = data.copy()
    if 'id' in df_out.columns:
        df_out.drop(columns=['id'], inplace=True)

    # missing value flag & domain features
    df_out['bmi_is_missing'] = df_out['bmi'].isna().astype(int)
    df_out['risk_factors_count'] = df_out['hypertension'] + df_out['heart_disease']
    df_out['is_senior'] = (df_out['age'] >= 60).astype(int)

    # ratios & interactions
    bmi_filled = df_out['bmi'].fillna(df_out['bmi'].median())
    df_out['glucose_bmi_ratio'] = df_out['avg_glucose_level'] / bmi_filled
    df_out['age_glucose_product'] = df_out['age'] * df_out['avg_glucose_level']

    # one-hot encoding
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    return pd.get_dummies(df_out, columns=cat_cols, drop_first=True)

def plot_all_confusion_matrices(cms: list, titles: list, output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ['No Stroke', 'Stroke']

    for ax, cm, title in zip(axes, cms, titles):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importance(model, output_path: str):
    importance = model.get_booster().get_score(importance_type='gain')
    imp_df = pd.DataFrame({
        'Feature': list(importance.keys()), 
        'Gain': list(importance.values())
    }).sort_values('Gain', ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(imp_df['Feature'], imp_df['Gain'], color='skyblue')
    plt.title('XGBoost Feature Importance (Gain Metric)', fontsize=12, fontweight='bold')
    plt.xlabel('Gain Value')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading dataset...")
    raw_df = pd.read_csv(DATASET_PATH)
    if 'id' in raw_df.columns:
        raw_df.drop(columns=['id'], inplace=True)

    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df_base = pd.get_dummies(raw_df.copy(), columns=cat_cols, drop_first=True)
    df_eng = preprocess_and_engineer(raw_df)

    X_base, y_base = df_base.drop(columns=['stroke']), df_base['stroke']
    X_eng, y_eng = df_eng.drop(columns=['stroke']), df_eng['stroke']

    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_base, y_base, test_size=0.2, random_state=SEED, stratify=y_base)
    X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(X_eng, y_eng, test_size=0.2, random_state=SEED, stratify=y_eng)

    scale_pos = (len(y_tr_e) - sum(y_tr_e)) / sum(y_tr_e)

    scenarios = [
        {
            'name': '1. Base',
            'data': (X_tr_b, X_te_b, y_tr_b, y_te_b),
            'model': XGBClassifier(random_state=SEED, eval_metric='logloss')
        },
        {
            'name': '2. Feature Engineering',
            'data': (X_tr_e, X_te_e, y_tr_e, y_te_e),
            'model': XGBClassifier(random_state=SEED, eval_metric='logloss', scale_pos_weight=scale_pos)
        },
        {
            'name': '3. Feature Engineering & Tuned XGBoost',
            'data': (X_tr_e, X_te_e, y_tr_e, y_te_e),
            'model': XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.5,
                scale_pos_weight=scale_pos, random_state=SEED, eval_metric='logloss'
            )
        }
    ]

    results, cms, titles = [], [], []

    print("Running experiments...")
    for sc in scenarios:
        X_tr, X_te, y_tr, y_te = sc['data']
        model = sc['model']
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:, 1]

        results.append({
            'Scenario': sc['name'],
            'ROC-AUC': roc_auc_score(y_te, probs),
            'F1-Score': f1_score(y_te, preds)
        })
        cms.append(confusion_matrix(y_te, preds))
        titles.append(sc['name'])

    print("\nEXPERIMENT RESULTS")
    print(pd.DataFrame(results).to_string(index=False))

    print("\nSaving plots...")
    plot_all_confusion_matrices(cms, titles, OUTPUT_MATRIKS_PATH)
    plot_feature_importance(scenarios[-1]['model'], OUTPUT_FEATURE_IMPORTANCE_PATH)
    print(f"Successfully saved:\n - {OUTPUT_MATRIKS_PATH}\n - {OUTPUT_FEATURE_IMPORTANCE_PATH}")

if __name__ == "__main__":
    main()
