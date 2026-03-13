# ============================================================
# 03_pyreason_manual.py
# Manual implementation of PyReason's annotated logic reasoning
# Replaces 02_define_rules.py + 03_run_inference.py
#
# Core concept: every predicate on every node has an annotation
# interval [lower, upper] representing confidence bounds.
# Rules fire when their body conditions are met, writing new
# annotations onto nodes. We repeat until nothing changes
# (bounded fixpoint computation).
# ============================================================

import pickle
import pandas as pd
from collections import defaultdict

# Load graph 
with open('graph/fraud_graph.pkl', 'rb') as f:
    G = pickle.load(f)

print(f"Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Annotation store 
# For every node, store a dict of predicate → [lower, upper]
# This is exactly what PyReason's interpretation object holds
annotations = defaultdict(dict)   # annotations[node][predicate] = [l, u]
rule_trace   = defaultdict(list)  # rule_trace[node] = list of rules that fired

#  Helper functions 
def set_annotation(node, predicate, lower, upper, fired_by=None):
    """Write annotation to a node. Only update if new bounds are stronger."""
    current = annotations[node].get(predicate, [0.0, 0.0])
    new_lower = max(current[0], lower)
    new_upper = max(current[1], upper)
    if [new_lower, new_upper] != current:
        annotations[node][predicate] = [new_lower, new_upper]
        if fired_by:
            rule_trace[node].append(fired_by)
        return True   # changed
    return False      # no change

def get_annotation(node, predicate):
    """Get annotation for a node's predicate. Default [0,0] if not set."""
    return annotations[node].get(predicate, [0.0, 0.0])

def meets_threshold(node, predicate, min_lower=0.0):
    """Check if a predicate is active (lower bound above threshold)."""
    return get_annotation(node, predicate)[0] >= min_lower

#Stamp initial facts 
# Read node attributes from graph and convert to annotation intervals
# This is what PyReason calls "adding facts"
print("\nStamping initial facts onto nodes...")

for node, data in G.nodes(data=True):

    if data.get('node_type') == 'transaction':
        # fraud_type — all our transactions are TRANSFER/CASH_OUT
        set_annotation(node, 'fraud_type', 1.0, 1.0)

        # large_amount
        v = float(data.get('is_large_amount', 0))
        set_annotation(node, 'large_amount', v, v)

        # round_amount
        v = float(data.get('is_round_amount', 0))
        set_annotation(node, 'round_amount', v, v)

        # high_velocity (account signal carried on transaction)
        v = float(data.get('is_high_velocity', 0))
        set_annotation(node, 'high_velocity', v, v)

        # repeat_amount
        v = float(data.get('is_repeat_amount', 0))
        set_annotation(node, 'repeat_amount', v, v)

    if data.get('node_type') == 'account':
        v = float(data.get('is_high_velocity', 0))
        set_annotation(node, 'high_velocity', v, v)

        v = float(data.get('is_repeat_amount', 0))
        set_annotation(node, 'repeat_amount', v, v)

print(f"Facts stamped on {len(annotations):,} nodes.")

# Fix high_velocity on account nodes — take the MAX across all their transactions
# (first-seen stamping misses accounts whose first tx didn't have the flag)
for node, data in G.nodes(data=True):
    if data.get('node_type') != 'account':
        continue
    # check all outgoing sends edges to find transactions this account sent
    max_hv = 0.0
    max_ra = 0.0
    for tx in G.successors(node):
        tx_data = G.nodes[tx]
        if tx_data.get('node_type') == 'transaction':
            max_hv = max(max_hv, float(tx_data.get('is_high_velocity', 0)))
            max_ra = max(max_ra, float(tx_data.get('is_repeat_amount', 0)))
    set_annotation(node, 'high_velocity', max_hv, max_hv)
    set_annotation(node, 'repeat_amount', max_ra, max_ra)

#  Rule definitions 
# Each rule is a function that:
#   - checks body conditions on relevant nodes
#   - calls set_annotation on head nodes
#   - returns True if anything changed (so fixpoint knows to keep going)
#
# Annotation intervals on rule heads match what we defined earlier:
#   [0.7, 0.9] = strong signal, not certain
#   [0.75, 0.95] = very strong signal
#   [0.8, 1.0] = near certain fraud
#   [0.6, 0.85] = moderate risk (propagation)

def rule_1_suspicious_amount(G):
    """
    R1a: large_amount(T) ∧ round_amount(T) → suspicious_amount(T): [0.8, 0.9]
    R1b: large_amount(T) → suspicious_amount(T): [0.6, 0.8]
    Large + round is stronger signal; large alone is weaker but still fires
    """
    changed = False
    for node, data in G.nodes(data=True):
        if data.get('node_type') != 'transaction':
            continue
        if (meets_threshold(node, 'large_amount', 1.0) and
                meets_threshold(node, 'round_amount', 1.0)):
            changed |= set_annotation(node, 'suspicious_amount', 0.8, 0.9,
                                      fired_by='R1a: large_amount ∧ round_amount')
        elif meets_threshold(node, 'large_amount', 1.0):
            changed |= set_annotation(node, 'suspicious_amount', 0.6, 0.8,
                                      fired_by='R1b: large_amount')
    return changed

def rule_2_suspicious_account(G):
    """
    R2a: high_velocity(A) ∧ repeat_amount(A) → suspicious_account(A): [0.8, 0.9]
    R2b: high_velocity(A) → suspicious_account(A): [0.6, 0.8]
    High velocity alone is enough to flag an account
    """
    changed = False
    for node, data in G.nodes(data=True):
        if data.get('node_type') != 'account':
            continue
        if (meets_threshold(node, 'high_velocity', 1.0) and
                meets_threshold(node, 'repeat_amount', 1.0)):
            changed |= set_annotation(node, 'suspicious_account', 0.8, 0.9,
                                      fired_by='R2a: high_velocity ∧ repeat_amount')
        elif meets_threshold(node, 'high_velocity', 1.0):
            changed |= set_annotation(node, 'suspicious_account', 0.6, 0.8,
                                      fired_by='R2b: high_velocity')
    return changed

def rule_3_suspicious_tx_amount(G):
    """
    R3: fraud_type(T) ∧ suspicious_amount(T) → suspicious(T): [0.7, 0.95]
    Lowered threshold to 0.6 so weaker suspicious_amount signal still fires
    """
    changed = False
    for node, data in G.nodes(data=True):
        if data.get('node_type') != 'transaction':
            continue
        if (meets_threshold(node, 'fraud_type', 1.0) and
                meets_threshold(node, 'suspicious_amount', 0.6)):
            changed |= set_annotation(node, 'suspicious', 0.7, 0.95,
                                      fired_by='R3: fraud_type ∧ suspicious_amount')
    return changed

def rule_4_suspicious_tx_account(G):
    """
    R4: fraud_type(T) ∧ suspicious_account(A) ∧ sends(A,T) → suspicious(T): [0.75, 0.95]
    Transaction is suspicious if sent by a suspicious account
    Graph traversal: walk the sends edge from account to transaction
    """
    changed = False
    for node, data in G.nodes(data=True):
        if data.get('node_type') != 'transaction':
            continue
        if not meets_threshold(node, 'fraud_type', 1.0):
            continue
        # Find the account that sent this transaction (predecessor via sends edge)
        for sender in G.predecessors(node):
            edge_data = G.edges[sender, node]
            if edge_data.get('relation') == 'sends':
                if meets_threshold(sender, 'suspicious_account', 0.7):
                    changed |= set_annotation(node, 'suspicious', 0.75, 0.95,
                                              fired_by=f'R4: fraud_type ∧ suspicious_account({sender[:8]})')
    return changed

def rule_5_fraud(G):
    """
    R5: suspicious(T) → fraud(T): [0.8, 1.0]
    Threshold lowered to 0.7 to match R3's output of [0.7, 0.95]
    """
    changed = False
    for node, data in G.nodes(data=True):
        if data.get('node_type') != 'transaction':
            continue
        if meets_threshold(node, 'suspicious', 0.7):
            changed |= set_annotation(node, 'fraud', 0.8, 1.0,
                                      fired_by='R5: suspicious → fraud')
    return changed

def rule_6_fraud_propagation(G):
    """
    R6: fraud(T) ∧ receives(T, A) → fraud_risk(A): [0.6, 0.85]
    Receiving account of a fraud transaction is at risk
    Graph propagation: fraud signal travels through receives edge
    """
    changed = False
    for node, data in G.nodes(data=True):
        if data.get('node_type') != 'transaction':
            continue
        if meets_threshold(node, 'fraud', 0.8):
            # Find accounts this transaction flows to (successor via receives edge)
            for receiver in G.successors(node):
                edge_data = G.edges[node, receiver]
                if edge_data.get('relation') == 'receives':
                    changed |= set_annotation(receiver, 'fraud_risk', 0.6, 0.85,
                                              fired_by=f'R6: fraud({node}) → fraud_risk')
    return changed

# ── 6. Bounded fixpoint computation ───────────────────────────
# This is the core of PyReason's engine.
#  Fire all rules repeatedly until nothing changes.
# Each iteration = one "timestep" in PyReason terminology.
# We cap at MAX_TIMESTEPS to guarantee termination.

MAX_TIMESTEPS = 10
rules = [
    rule_1_suspicious_amount,
    rule_2_suspicious_account,
    rule_3_suspicious_tx_amount,
    rule_4_suspicious_tx_account,
    rule_5_fraud,
    rule_6_fraud_propagation,
]

print("\nRunning bounded fixpoint computation...")
for t in range(MAX_TIMESTEPS):
    any_changed = False
    for rule in rules:
        any_changed |= rule(G)

    fired = sum(1 for rule in rules if rule.__name__)
    print(f"  Timestep {t+1}: {'changes made' if any_changed else 'no changes — converged'}")

    if not any_changed:
        print(f"  Fixpoint reached at timestep {t+1}")
        break

#  Results extraction
flagged_tx = []
flagged_accounts = []

for node, data in G.nodes(data=True):
    if data.get('node_type') == 'transaction':
        fraud_ann = get_annotation(node, 'fraud')
        if fraud_ann[0] > 0:
            flagged_tx.append({
                'node'        : node,
                'fraud_lower' : fraud_ann[0],
                'fraud_upper' : fraud_ann[1],
                'annotations' : {k: v for k, v in annotations[node].items()
                                 if v[0] > 0},
                'rules_fired' : rule_trace[node]
            })

    if data.get('node_type') == 'account':
        risk_ann = get_annotation(node, 'fraud_risk')
        if risk_ann[0] > 0:
            flagged_accounts.append({
                'node'             : node,
                'fraud_risk_lower' : risk_ann[0],
                'fraud_risk_upper' : risk_ann[1],
            })

df_flagged  = pd.DataFrame(flagged_tx)
df_risk     = pd.DataFrame(flagged_accounts)

print(f"\nResults:")
print(f"  Transactions flagged as fraud : {len(df_flagged):,}")
print(f"  Accounts flagged as fraud_risk: {len(df_risk):,}")

# Merge with ground truth 
df_sample = pd.read_csv('data/paysim_sample_features.csv')
df_sample['tx_id'] = ['TX_' + str(i) for i in df_sample.index]

df_eval = df_sample[['tx_id', 'isFraud', 'amount', 'type',
                      'is_large_amount', 'is_high_velocity',
                      'is_round_amount', 'is_repeat_amount']].copy()

flagged_set = set(df_flagged['node'].tolist()) if len(df_flagged) > 0 else set()
df_eval['pyreason_flagged'] = df_eval['tx_id'].isin(flagged_set).astype(int)

# Rule trace (explainability)
print("\n=== Rule trace for first 3 flagged transactions ===")
for _, row in df_flagged.head(3).iterrows():
    print(f"\n  {row['node']}:")
    for pred, bounds in row['annotations'].items():
        print(f"    {pred:<25} [{bounds[0]:.2f}, {bounds[1]:.2f}]")
    print(f"    Rules fired: {row['rules_fired']}")

# Save results 
df_eval.to_csv('data/inference_results.csv', index=False)
df_flagged.drop(columns=['annotations','rules_fired']).to_csv(
    'data/flagged_transactions.csv', index=False)
df_risk.to_csv('data/flagged_accounts.csv', index=False)

print("\nSaved:")
print("  data/inference_results.csv")
print("  data/flagged_transactions.csv")
print("  data/flagged_accounts.csv")

print("\nRun 04_evaluate.py next.")
