# run_inference.py
# Phase 4 — PyReason Inference
# Load facts + rules → run reasoning → extract results

import pickle
import pandas as pd
import pyreason as pr

# Load graph
with open('graph/fraud_graph.pkl', 'rb') as f:
    G = pickle.load(f)

print(f"Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Load rules from define_rules 
# We import the setup function so inference is self-contained and we don't duplicate fact/rule definitions
exec(open('02_define_rules.py').read())

# Run inference 
# timesteps = how many rounds of rule firing PyReason runs
# 1 timestep  → only direct facts fire (R1, R2)
# 2 timesteps → chained rules fire (R3, R4, R5)
# 3 timesteps → propagation fires (R6 — fraud_risk on accounts)
# We use 3 to get the full chain

print("\nRunning PyReason inference (3 timesteps)...")
interpretation = pr.reason(timesteps=3)
print("Inference complete.")

#  Extracting results 
# interpretation contains annotation tables for every predicate
# We care about:
#   fraud(TX)      → which transactions were flagged
#   fraud_risk(A)  → which accounts were flagged via propagation

# --- Fraud transactions ---
fraud_results = interpretation.get_component_interpretation('fraud')

flagged_tx = []
for node, bounds in fraud_results.items():
    lower, upper = bounds
    if lower > 0:   # lower bound > 0 means the rule actually fired
        flagged_tx.append({
            'node'       : node,
            'fraud_lower': lower,
            'fraud_upper': upper
        })

df_flagged = pd.DataFrame(flagged_tx)
print(f"\nTransactions flagged as fraud: {len(df_flagged):,}")

# --- Fraud risk accounts ---
risk_results = interpretation.get_component_interpretation('fraud_risk')

flagged_accounts = []
for node, bounds in risk_results.items():
    lower, upper = bounds
    if lower > 0:
        flagged_accounts.append({
            'node'            : node,
            'fraud_risk_lower': lower,
            'fraud_risk_upper': upper
        })

df_risk = pd.DataFrame(flagged_accounts)
print(f"Accounts flagged as fraud_risk: {len(df_risk):,}")

# Merge with ground truth 

#  sample features CSV (has ground truth isFraud labels)
df_sample = pd.read_csv('data/paysim_sample_features.csv')

# Create TX node IDs to match what we built in build_graph.py
df_sample['tx_id'] = ['TX_' + str(i) for i in df_sample.index]

# Merge flagged transactions with ground truth
df_eval = df_sample[['tx_id', 'isFraud', 'amount', 'type',
                      'is_large_amount', 'is_high_velocity',
                      'is_round_amount', 'is_repeat_amount']].copy()

df_eval['pyreason_flagged'] = df_eval['tx_id'].isin(
    df_flagged['node'] if len(df_flagged) > 0 else []
).astype(int)

# Quick results preview
print("\n=== Flagged transactions sample ===")
if len(df_flagged) > 0:
    preview = df_eval[df_eval['pyreason_flagged'] == 1][
        ['tx_id','isFraud','amount','type','is_large_amount',
         'is_high_velocity','is_round_amount','is_repeat_amount']
    ].head(10)
    print(preview.to_string(index=False))
else:
    print("No transactions flagged — check rule thresholds in 02_define_rules.py")

# Rule trace (explainability) 
# For each flagged transaction, see WHICH rules fired

# This is PyReason's killer feature vs black-box ML
print("\n=== Rule trace for first 3 flagged transactions ===")
if len(df_flagged) > 0:
    for tx in df_flagged['node'].head(3).tolist():
        print(f"\n  {tx}:")
        trace = interpretation.get_node_interpretation(tx)
        for predicate, bounds in trace.items():
            if bounds[0] > 0:
                print(f"    {predicate:<25} [{bounds[0]:.2f}, {bounds[1]:.2f}]")

# Saving results 
df_eval.to_csv('data/inference_results.csv', index=False)
df_flagged.to_csv('data/flagged_transactions.csv', index=False)
df_risk.to_csv('data/flagged_accounts.csv', index=False)

print("\nSaved:")
print("  data/inference_results.csv     ← full eval table")
print("  data/flagged_transactions.csv  ← PyReason flagged txs")
print("  data/flagged_accounts.csv      ← PyReason flagged accounts")
print("\nPhase 4 complete. Run 04_evaluate.py next.")