# 🔍 FraudLens — Annotated Logic Fraud Detection with PyReason

> Symbolic reasoning over transaction knowledge graphs. Every fraud flag has a full audit trail. No black boxes.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Framework](https://img.shields.io/badge/Reasoning-PyReason-purple)](https://github.com/lab-v2/pyreason)
[![Dataset](https://img.shields.io/badge/Dataset-PaySim-green)](https://www.kaggle.com/datasets/ealaxi/paysim1)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)](STREAMLIT_LINK_HERE)

---

## Live Demo

👉 **[Try FraudLens on Streamlit](STREAMLIT_LINK_HERE)**

Upload `paysim_demo.csv` to see the reasoning engine flag transactions in real time with full rule traces.

---

## What is this?

FraudLens implements **annotated logic reasoning** over a transaction knowledge graph, inspired by [PyReason](https://github.com/lab-v2/pyreason) — a framework for graph-based symbolic AI developed by Prof. Paulo Shakarian at Arizona State University.

Instead of training a machine learning model, FraudLens encodes fraud domain knowledge as **logical rules with confidence intervals**. The reasoning engine fires these rules over a knowledge graph of accounts and transactions, propagating fraud signals through connected nodes until it reaches a fixpoint.

---

## Why not just use a ML model?

| | ML Models | FraudLens (PyReason-style) |
|---|---|---|
| Requires labeled training data | ✅ Yes | ❌ No |
| Explainable per-flag reason | ❌ No | ✅ Yes — full rule trace |
| Human-tunable thresholds | ❌ Retrain required | ✅ Edit rules directly |
| Graph propagation | ❌ Flat features only | ✅ Fraud spreads through edges |
| Auditable for compliance | ❌ Black box | ✅ Every flag is traceable |

In real FinTech, clean fraud labels are often unavailable or delayed. PyReason-style reasoning lets analysts encode what they *know* about fraud behavior and get immediate, explainable results.

---

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **FraudLens (PyReason)** | 0.06 | 0.13 | 0.08 | 0.54 |
| Logistic Regression | 0.50 | 0.38 | 0.43 | 0.68 |
| Random Forest | 0.50 | 0.38 | 0.43 | 0.68 |

**FraudLens doesn't win on raw F1 — and that's expected.** The rules were written from domain knowledge alone, with no exposure to training labels. The ML baselines were trained on the same data they're evaluated on.

The real advantage is **explainability**. Here's a sample rule trace for a flagged transaction:

```
TX_28:
  fraud_type          [1.00, 1.00]  ← TRANSFER/CASH_OUT
  large_amount        [1.00, 1.00]  ← above 95th percentile
  round_amount        [1.00, 1.00]  ← no cents, scripted behavior
  suspicious_amount   [0.80, 0.90]  ← R1a fired
  suspicious          [0.70, 0.95]  ← R3 fired
  fraud               [0.80, 1.00]  ← R5 fired

Rules fired: R1a → R3 → R5
```

A compliance team can audit exactly why any transaction was flagged. No ML model gives you that.

---

## Knowledge Graph Structure

```
Account (nameOrig)
    │
    │  sends
    ▼
Transaction (TX_i)  ── attributes: fraud_type, large_amount, round_amount, ...
    │
    │  receives
    ▼
Account (nameDest)
```

Fraud signals propagate through the graph — a flagged transaction also raises the `fraud_risk` score of its receiving account.

---

## Rules

| Rule | Logic | Annotation |
|---|---|---|
| R1a | large_amount ∧ round_amount → suspicious_amount | [0.8, 0.9] |
| R1b | large_amount → suspicious_amount | [0.6, 0.8] |
| R2a | high_velocity ∧ repeat_amount → suspicious_account | [0.8, 0.9] |
| R2b | high_velocity → suspicious_account | [0.6, 0.8] |
| R3 | fraud_type ∧ suspicious_amount → suspicious | [0.7, 0.95] |
| R4 | fraud_type ∧ suspicious_account ∧ sends(A,T) → suspicious | [0.7, 0.95] |
| R5 | suspicious → fraud | [0.8, 1.0] |
| R6 | fraud ∧ receives(T,A) → fraud_risk | [0.6, 0.85] |

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/MuktaPatil/pyreason-fraud-detection.git
cd pyreason-fraud-detection
```

### 2. Create environment
```bash
conda create -n fraud_pyreason python=3.11
conda activate fraud_pyreason
pip install pandas numpy networkx scikit-learn matplotlib streamlit
```

### 3. Get the data
Download [PaySim from Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place the CSV in `data/`.

Then in the Colab EDA notebook, export the filtered file:
```python
df_rel = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
df_rel.to_csv('paysim_filtered.csv', index=False)
```

### 4. Run the pipeline
```bash
# Step 1 — Build knowledge graph
python 01_build_graph.py

# Step 2 — Run PyReason-style inference
python 03_pyreason_manual.py

# Step 3 — Evaluate vs ML baselines
python 04_evaluate.py
```

### 5. Run the Streamlit demo
```bash
streamlit run app.py
```
Upload `paysim_demo.csv` (a 5000-row demo sample) to see the live rule traces.

---

## Project Structure

```
pyreason-fraud-detection/
│
├── 01_build_graph.py         # Builds NetworkX knowledge graph from PaySim
├── 03_pyreason_manual.py     # Annotated logic reasoning engine (6 rules)
├── 04_evaluate.py            # Evaluation vs Logistic Regression + Random Forest
├── app.py                    # Streamlit demo app
├── data/                     # Place your CSV files here (not tracked by git)
└── README.md
```

---

## Built With

- [PyReason](https://github.com/lab-v2/pyreason) — annotated logic reasoning framework
- [NetworkX](https://networkx.org) — knowledge graph construction
- [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — synthetic mobile money transaction dataset
- [Streamlit](https://streamlit.io) — interactive demo

---

*Built as part of a graduate project at Syracuse University's iSchool.*
