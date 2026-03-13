# ============================================================
# 04_evaluate.py
# Phase 5 — Evaluation & Comparison
# PyReason results vs ML baselines (Logistic Regression + Random Forest)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load inference results ─────────────────────────────────
df = pd.read_csv('data/inference_results.csv')
print(f"Loaded {len(df):,} transactions")
print(f"Fraud cases: {df['isFraud'].sum():,}")
print(f"PyReason flagged: {df['pyreason_flagged'].sum():,}")

# ── 2. Define features for ML baselines ───────────────────────
# Same features PyReason used — fair comparison
FEATURES = ['is_large_amount', 'is_high_velocity',
            'is_round_amount', 'is_repeat_amount']
X = df[FEATURES].values
y = df['isFraud'].values

# ── 3. Train/test split ───────────────────────────────────────
# Stratified split — guarantees fraud cases in both train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# PyReason predictions aligned to test rows
y_pr = df.loc[idx_test, 'pyreason_flagged'].values

print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")
print(f"Fraud in test: {y_test.sum():,}")

# ── 4. Logistic Regression baseline ──────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(X_train_sc, y_train)
y_lr = lr.predict(X_test_sc)

# ── 5. Random Forest baseline ─────────────────────────────────
rf = RandomForestClassifier(
    n_estimators    = 100,
    class_weight    = 'balanced',
    random_state    = 42,
    n_jobs          = -1
)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)

# ── 6. Compute metrics ────────────────────────────────────────
def metrics(y_true, y_pred, name):
    return {
        'Model'    : name,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall'   : recall_score(y_true, y_pred, zero_division=0),
        'F1'       : f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC'  : roc_auc_score(y_true, y_pred) if y_pred.sum() > 0 else 0.0
    }

results = pd.DataFrame([
    metrics(y_test, y_pr, 'PyReason'),
    metrics(y_test, y_lr, 'Logistic Regression'),
    metrics(y_test, y_rf, 'Random Forest'),
])
results = results.set_index('Model').round(4)

print("\n=== Evaluation Results ===")
print(results.to_string())

# ── 7. Explainability note ────────────────────────────────────
# PyReason's real advantage isn't always raw F1 — it's that
# every flagged transaction has a traceable rule chain.
# Print a reminder of this for the writeup.
print("""
=== PyReason vs ML — Key Distinction ===
ML models are black boxes — they give a fraud probability
with no explanation of why.

PyReason gives a rule trace per flagged transaction:
  TX_42:
    fraud_type        [1.00, 1.00]  ← TRANSFER/CASH_OUT
    large_amount      [1.00, 1.00]  ← above 95th pct
    round_amount      [1.00, 1.00]  ← no cents, scripted
    suspicious_amount [0.70, 0.90]  ← R1 fired
    suspicious        [0.75, 0.95]  ← R3 fired
    fraud             [0.80, 1.00]  ← R5 fired

Every flag is auditable. That matters in FinTech.
""")

# ── 8. Visualizations ─────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 8a. Metrics bar chart
ax1 = fig.add_subplot(gs[0, :2])
results[['Precision','Recall','F1']].plot(
    kind='bar', ax=ax1,
    color=['#3B8BD4','#E24B4A','#1D9E75'], alpha=0.85
)
ax1.set_title('Precision / Recall / F1 by model')
ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=20)
ax1.legend(loc='upper right')
ax1.set_ylim(0, 1.1)
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.2f', fontsize=8, padding=2)

# 8b. ROC-AUC bar chart
ax2 = fig.add_subplot(gs[0, 2])
roc_vals = results['ROC-AUC']
bars = ax2.bar(roc_vals.index, roc_vals.values,
               color=['#7F77DD','#3B8BD4','#1D9E75'], alpha=0.85)
ax2.set_title('ROC-AUC by model')
ax2.set_ylim(0, 1.1)
ax2.tick_params(axis='x', rotation=20)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f'{bar.get_height():.2f}',
             ha='center', fontsize=9)

plt.suptitle('PyReason Fraud Detection — Evaluation', fontsize=14, y=1.01)
plt.savefig('data/evaluation_results.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: data/evaluation_results.png")
print("\nPhase 5 complete!")