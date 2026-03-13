# build_graph.py
# Project Phase 2 — Knowledge Graph Construction
# Task : Reads paysim_filtered.csv → builds a NetworkX graph that PyReason can consume


import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os

CHUNK_SIZE = 50_000

df = pd.read_csv('data/paysim_filtered.csv', nrows=CHUNK_SIZE)
print(f"Loaded {len(df):,} transactions")
print(f"Columns: {list(df.columns)}")

# Sampling for for project 
# Full PaySim is huge, we work with a manageable sample while
# building the project. We stratify so fraud cases are preserved.
SAMPLE_SIZE = 5000

fraud = df[df['isFraud'] == 1]
legit = df[df['isFraud'] == 0].sample(
    n=min(SAMPLE_SIZE - len(fraud), len(df[df['isFraud'] == 0])),
    random_state=42
)
df_sample = pd.concat([fraud, legit]).reset_index(drop=True)
print(f"\nWorking sample: {len(df_sample):,} transactions")
print(f"  Fraud: {df_sample['isFraud'].sum():,}")
print(f"  Legit: {(df_sample['isFraud'] == 0).sum():,}")


#  Compute thresholds from EDA ────────────────────────────
# These come from our EDA findings — do NOT use balance columns

LARGE_AMOUNT_THRESHOLD = df['amount'].quantile(0.95)
VELOCITY_THRESHOLD     = 3   # transactions in a single step (hour)

print(f"\nThresholds:")
print(f"  large_amount  → amount > ${LARGE_AMOUNT_THRESHOLD:,.2f}")
print(f"  high_velocity → tx per hour > {VELOCITY_THRESHOLD}")

# Feature engineering section
# Derive predicates per ACCOUNT (based on their transaction history)
# and per TRANSACTION (based on the row itself)

#  Account-level features 

# a. High velocity: max transactions by this account in any single step
velocity = (df_sample.groupby(['nameOrig', 'step'])
              .size()
              .reset_index(name='tx_in_step'))
max_velocity = (velocity.groupby('nameOrig')['tx_in_step']
                  .max()
                  .reset_index(name='max_hourly_tx'))

# b. Repeat amount: does this account send the same amount 2+ times?
repeat_amt = (df_sample.groupby(['nameOrig', 'amount'])
                .size()
                .reset_index(name='count'))
repeat_accounts = set(
    repeat_amt[repeat_amt['count'] > 1]['nameOrig']
)

# Merge account features back onto transactions
df_sample = df_sample.merge(max_velocity, on='nameOrig', how='left')
df_sample['max_hourly_tx'] = df_sample['max_hourly_tx'].fillna(1)

# --- Transaction-level features ---
# c. Large amounts -- suspicious
df_sample['is_large_amount'] = (
    df_sample['amount'] > LARGE_AMOUNT_THRESHOLD
).astype(int)

# d. Round amount (no cents ?= scripted behavior)
df_sample['is_round_amount'] = (
    df_sample['amount'] % 1 == 0
).astype(int)

# e. High velocity (account-level signal, attached to transaction)
df_sample['is_high_velocity'] = (
    df_sample['max_hourly_tx'] > VELOCITY_THRESHOLD
).astype(int)

# f. Repeat amount flag (account-level, attached to transaction)
df_sample['is_repeat_amount'] = (
    df_sample['nameOrig'].isin(repeat_accounts)
).astype(int)

print(f"\nFeature summary (% of transactions):")
for col in ['is_large_amount','is_round_amount','is_high_velocity','is_repeat_amount']:
    pct = df_sample[col].mean() * 100
    fraud_pct = df_sample[df_sample['isFraud']==1][col].mean() * 100
    print(f"  {col:<22} all={pct:.1f}%  fraud={fraud_pct:.1f}%")

# Building  the knowledge graph 
'''Node types:
  - Account nodes  → nameOrig and nameDest values
   - Transaction nodes → unique transaction ID we create

# Edge types:
  - sends(Account, Transaction)    → account originated this tx
  - receives(Account, Transaction) → account received this tx
'''
G = nx.DiGraph()

print("\nBuilding graph...")

for idx, row in df_sample.iterrows():
    tx_id     = f"TX_{idx}"
    orig_id   = row['nameOrig']
    dest_id   = row['nameDest']

    # Adding transaction node 
    # Attributes = our derived predicates (stored as 0/1)
    # PyReason to read these to set initial annotation intervals
    G.add_node(tx_id, node_type='transaction',
               is_large_amount   = int(row['is_large_amount']),
               is_round_amount   = int(row['is_round_amount']),
               is_high_velocity  = int(row['is_high_velocity']),
               is_repeat_amount  = int(row['is_repeat_amount']),
               fraud_type        = 1,   # already filtered to TRANSFER/CASH_OUT
               is_fraud          = int(row['isFraud']),   # ground truth label
               amount            = float(row['amount']),
               step              = int(row['step']),
               tx_type           = row['type'])

    # Adding account nodes --new
    # Only add if not already in graph (accounts appear in many txs)
    if orig_id not in G:
        G.add_node(orig_id, node_type='account',
                   is_high_velocity = int(row['is_high_velocity']),
                   is_repeat_amount = int(row['is_repeat_amount']))

    if dest_id not in G:
        G.add_node(dest_id, node_type='account',
                   is_high_velocity = 0,
                   is_repeat_amount = 0)

    #  Adding edges 
    G.add_edge(orig_id, tx_id, relation='sends')
    G.add_edge(tx_id, dest_id, relation='receives')

#  Graph stats 
account_nodes = [n for n,d in G.nodes(data=True) if d.get('node_type')=='account']
tx_nodes      = [n for n,d in G.nodes(data=True) if d.get('node_type')=='transaction']

print(f"\nGraph built:")
print(f"  Total nodes : {G.number_of_nodes():,}")
print(f"  Account nodes: {len(account_nodes):,}")
print(f"  Transaction nodes: {len(tx_nodes):,}")
print(f"  Total edges : {G.number_of_edges():,}")
print(f"  Is directed : {G.is_directed()}")

# sanity check — every tx node should have exactly 2 edges
degrees = [G.degree(n) for n in tx_nodes]
print(f"  Tx node degree range: {min(degrees)} – {max(degrees)} (should be 2)")

# Save graph 
os.makedirs('graph', exist_ok=True)

# Save as pickle (preserves all Python attributes cleanly)
with open('graph/fraud_graph.pkl', 'wb') as f:
    pickle.dump(G, f)

# Also save as GraphML (human-readable, good for inspection)
nx.write_graphml(G, 'graph/fraud_graph.graphml')

# Save sample df with features for later evaluation
df_sample.to_csv('data/paysim_sample_features.csv', index=False)

print(f"\nSaved:")
print(f"  graph/fraud_graph.pkl      ← PyReason will load this")
print(f"  graph/fraud_graph.graphml  ← open in Gephi to visualize")
print(f"  data/paysim_sample_features.csv ← features for evaluation")
# print(f"\nPhase 2 complete. Run define_rules.py next.")