# define_rules.py
# Phase 3 — Knowledge Base Design
# Load the graph → stamp initial facts → define PyReason rules

import pickle
import pyreason as pr

# Load graph 
with open('graph/fraud_graph.pkl', 'rb') as f:
    G = pickle.load(f)

print(f"Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Load g into PyReason 
pr.load_graph(G)

# Stamp initial facts (annotations) onto nodes
# Read the attributes we stored in build_graph.py and convert them into PyReason annotation intervals.

# The pattern is simple:
#   attribute == 1  → annotation [1, 1]  (definite fact)
#   attribute == 0  → annotation [0, 0]  (definitely not)

for node, data in G.nodes(data=True):

    if data.get('node_type') == 'transaction':

        # fraud_type — already filtered to TRANSFER/CASH_OUT
        pr.add_fact(pr.Fact(
            name         = 'fraud_type_fact',
            node         = node,
            component    = 'fraud_type',
            lower_bound  = 1.0,
            upper_bound  = 1.0,
            static       = True
        ))

        # large_amount
        val = float(data.get('is_large_amount', 0))
        pr.add_fact(pr.Fact(
            name         = f'large_amount_fact_{node}',
            node         = node,
            component    = 'large_amount',
            lower_bound  = val,
            upper_bound  = val,
            static       = True
        ))

        # round_amount
        val = float(data.get('is_round_amount', 0))
        pr.add_fact(pr.Fact(
            name         = f'round_amount_fact_{node}',
            node         = node,
            component    = 'round_amount',
            lower_bound  = val,
            upper_bound  = val,
            static       = True
        ))

    if data.get('node_type') == 'account':

        # high_velocity
        val = float(data.get('is_high_velocity', 0))
        pr.add_fact(pr.Fact(
            name         = f'high_velocity_fact_{node}',
            node         = node,
            component    = 'high_velocity',
            lower_bound  = val,
            upper_bound  = val,
            static       = True
        ))

        # repeat_amount
        val = float(data.get('is_repeat_amount', 0))
        pr.add_fact(pr.Fact(
            name         = f'repeat_amount_fact_{node}',
            node         = node,
            component    = 'repeat_amount',
            lower_bound  = val,
            upper_bound  = val,
            static       = True
        ))

print("Facts stamped onto all nodes.")

'''Define rules 
# Rules fire in order. Each one produces a new predicate that
# the next rule can build on. This is rule chaining.
#
# Reading a rule:
#   HEAD <- BODY
#   "If BODY is true with this confidence, conclude HEAD"
#
# The annotation on the head [l, u] is what gets written onto
# the node when the rule fires.
''' 
# Rule 1: suspicious_amount ─────────────────────────────────
# A transaction is amount-suspicious if it is large AND round
# (large alone is common; large + round = scripted behavior)
r1 = pr.Rule(
    rule_text   = 'suspicious_amount(x):[0.7,0.9] <- large_amount(x):[1,1], round_amount(x):[1,1]',
    rule_name   = 'suspicious_amount_rule'
)
pr.add_rule(r1)

# ── Rule 2: suspicious_account ────────────────────────────────
# An account is suspicious if it has high velocity AND repeats amounts
r2 = pr.Rule(
    rule_text   = 'suspicious_account(x):[0.7,0.9] <- high_velocity(x):[1,1], repeat_amount(x):[1,1]',
    rule_name   = 'suspicious_account_rule'
)
pr.add_rule(r2)

# ── Rule 3: suspicious_transaction ────────────────────────────
# A transaction is suspicious if:
#   - it is of fraud type (TRANSFER or CASH_OUT)
#   - AND either its amount is suspicious OR its origin account is suspicious
#
# Graph traversal: sends(account, transaction) edge connects them
r3 = pr.Rule(
    rule_text   = 'suspicious(y):[0.75,0.95] <- fraud_type(y):[1,1], suspicious_amount(y):[0.7,0.9]',
    rule_name   = 'suspicious_tx_amount_rule'
)
pr.add_rule(r3)

r4 = pr.Rule(
    rule_text   = 'suspicious(y):[0.75,0.95] <- fraud_type(y):[1,1], suspicious_account(x):[0.7,0.9], sends(x,y):[1,1]',
    rule_name   = 'suspicious_tx_account_rule'
)
pr.add_rule(r4)

# ── Rule 4: fraud ─────────────────────────────────────────────
# A transaction is fraud if it is suspicious with high enough confidence
r5 = pr.Rule(
    rule_text   = 'fraud(x):[0.8,1.0] <- suspicious(x):[0.75,0.95]',
    rule_name   = 'fraud_rule'
)
pr.add_rule(r5)

# ── Rule 5: fraud propagation ─────────────────────────────────
# If a transaction is fraud, flag the receiving account as at-risk
# This is the graph propagation that makes PyReason powerful —
# fraud signal travels through edges to connected nodes
r6 = pr.Rule(
    rule_text   = 'fraud_risk(y):[0.6,0.85] <- fraud(x):[0.8,1.0], receives(x,y):[1,1]',
    rule_name   = 'fraud_propagation_rule'
)
pr.add_rule(r6)

print("\nRules defined:")
rules = [
    ("R1", "suspicious_amount",    "large_amount ∧ round_amount → suspicious_amount"),
    ("R2", "suspicious_account",   "high_velocity ∧ repeat_amount → suspicious_account"),
    ("R3", "suspicious (amount)",  "fraud_type ∧ suspicious_amount → suspicious"),
    ("R4", "suspicious (account)", "fraud_type ∧ suspicious_account ∧ sends → suspicious"),
    ("R5", "fraud",                "suspicious → fraud"),
    ("R6", "fraud_risk",           "fraud ∧ receives → fraud_risk (propagation)"),
]
for rid, name, desc in rules:
    print(f"  {rid}  {name:<22} {desc}")

# print("\nPhase 3 complete. Run run_inference.py next.")pyt