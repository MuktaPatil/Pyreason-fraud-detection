# Streamlit code for the app

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
from collections import defaultdict


st.set_page_config(
    page_title="FraudLens",
    page_icon="🔍",
    layout="wide"
)


st.title("🔍 FraudLens")
st.caption("Annotated Logic Fraud Detection using PyReason-style Reasoning")
st.markdown("---")


st.sidebar.header("Settings")
large_amount_pct = st.sidebar.slider(
    "Large amount threshold (percentile)", 80, 99, 95)
velocity_threshold = st.sidebar.slider(
    "High velocity threshold (tx/hour)", 2, 10, 3)
show_legit = st.sidebar.checkbox("Show non-flagged transactions too", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown(
    "FraudLens implements annotated logic reasoning over a transaction "
    "knowledge graph. Every fraud flag comes with a full rule trace — "
    "no black boxes."
)

#  File upload 
st.subheader("Upload transaction data")
uploaded = st.file_uploader(
    "Upload a PaySim-format CSV (needs: step, type, amount, nameOrig, nameDest, isFraud)",
    type="csv"
)

if uploaded is None:
    st.info("Upload a CSV to get started. You can use paysim_filtered.csv from your data/ folder.")
    st.stop()


df_raw = pd.read_csv(uploaded, nrows=2000)
required = ['step','type','amount','nameOrig','nameDest']
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

df = df_raw[df_raw['type'].isin(['TRANSFER','CASH_OUT'])].reset_index(drop=True)
has_labels = 'isFraud' in df.columns

st.success(f"Loaded {len(df):,} TRANSFER/CASH_OUT transactions"
           + (f" | {df['isFraud'].sum():,} labeled fraud" if has_labels else ""))

# ── Feature engineering ───────────────────────────────────────
LARGE_AMOUNT_THRESHOLD = df['amount'].quantile(large_amount_pct / 100)

velocity = (df.groupby(['nameOrig','step'])
              .size()
              .reset_index(name='tx_in_step'))
max_vel = (velocity.groupby('nameOrig')['tx_in_step']
             .max()
             .reset_index(name='max_hourly_tx'))

repeat_amt = (df.groupby(['nameOrig','amount'])
                .size()
                .reset_index(name='count'))
repeat_accounts = set(repeat_amt[repeat_amt['count'] > 1]['nameOrig'])

df = df.merge(max_vel, on='nameOrig', how='left')
df['max_hourly_tx']    = df['max_hourly_tx'].fillna(1)
df['is_large_amount']  = (df['amount'] > LARGE_AMOUNT_THRESHOLD).astype(int)
df['is_round_amount']  = (df['amount'] % 1 == 0).astype(int)
df['is_high_velocity'] = (df['max_hourly_tx'] > velocity_threshold).astype(int)
df['is_repeat_amount'] = df['nameOrig'].isin(repeat_accounts).astype(int)
df['tx_id']            = ['TX_' + str(i) for i in df.index]

#  Build graph 
@st.cache_data(show_spinner="Building knowledge graph...")
def build_graph(_df):
    G = nx.DiGraph()
    for idx, row in _df.iterrows():
        tx_id   = row['tx_id']
        orig_id = row['nameOrig']
        dest_id = row['nameDest']
        G.add_node(tx_id, node_type='transaction',
                   is_large_amount  = int(row['is_large_amount']),
                   is_round_amount  = int(row['is_round_amount']),
                   is_high_velocity = int(row['is_high_velocity']),
                   is_repeat_amount = int(row['is_repeat_amount']))
        if orig_id not in G:
            G.add_node(orig_id, node_type='account',
                       is_high_velocity=int(row['is_high_velocity']),
                       is_repeat_amount=int(row['is_repeat_amount']))
        if dest_id not in G:
            G.add_node(dest_id, node_type='account',
                       is_high_velocity=0, is_repeat_amount=0)
        G.add_edge(orig_id, tx_id, relation='sends')
        G.add_edge(tx_id, dest_id, relation='receives')
    return G

G = build_graph(df)

# Reasoning engine 
annotations = defaultdict(dict)
rule_trace   = defaultdict(list)

def set_ann(node, pred, l, u, fired_by=None):
    cur = annotations[node].get(pred, [0.0, 0.0])
    nl, nu = max(cur[0], l), max(cur[1], u)
    if [nl, nu] != cur:
        annotations[node][pred] = [nl, nu]
        if fired_by:
            rule_trace[node].append(fired_by)
        return True
    return False

def get_ann(node, pred):
    return annotations[node].get(pred, [0.0, 0.0])

def meets(node, pred, min_l=0.0):
    return get_ann(node, pred)[0] >= min_l

# Stamp facts
for node, data in G.nodes(data=True):
    if data.get('node_type') == 'transaction':
        set_ann(node, 'fraud_type',    1.0, 1.0)
        v = float(data.get('is_large_amount', 0))
        set_ann(node, 'large_amount',  v, v)
        v = float(data.get('is_round_amount', 0))
        set_ann(node, 'round_amount',  v, v)

# Fix account-level features
for node, data in G.nodes(data=True):
    if data.get('node_type') != 'account':
        continue
    max_hv = max_ra = 0.0
    for tx in G.successors(node):
        if G.nodes[tx].get('node_type') == 'transaction':
            max_hv = max(max_hv, float(G.nodes[tx].get('is_high_velocity', 0)))
            max_ra = max(max_ra, float(G.nodes[tx].get('is_repeat_amount', 0)))
    set_ann(node, 'high_velocity', max_hv, max_hv)
    set_ann(node, 'repeat_amount', max_ra, max_ra)

# Rules: they rule
def run_rules(G):
    changed = False
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'transaction':
            if meets(node,'large_amount',1.0) and meets(node,'round_amount',1.0):
                changed |= set_ann(node,'suspicious_amount',0.8,0.9,'R1a: large_amount ∧ round_amount')
            elif meets(node,'large_amount',1.0):
                changed |= set_ann(node,'suspicious_amount',0.6,0.8,'R1b: large_amount')
        if data.get('node_type') == 'account':
            if meets(node,'high_velocity',1.0) and meets(node,'repeat_amount',1.0):
                changed |= set_ann(node,'suspicious_account',0.8,0.9,'R2a: high_velocity ∧ repeat_amount')
            elif meets(node,'high_velocity',1.0):
                changed |= set_ann(node,'suspicious_account',0.6,0.8,'R2b: high_velocity')

    for node, data in G.nodes(data=True):
        if data.get('node_type') != 'transaction':
            continue
        if meets(node,'fraud_type',1.0) and meets(node,'suspicious_amount',0.6):
            changed |= set_ann(node,'suspicious',0.7,0.95,'R3: fraud_type ∧ suspicious_amount')
        for sender in G.predecessors(node):
            if G.edges[sender,node].get('relation') == 'sends':
                if meets(sender,'suspicious_account',0.6):
                    changed |= set_ann(node,'suspicious',0.7,0.95,
                                       f'R4: suspicious_account({sender[:8]})')
        if meets(node,'suspicious',0.7):
            changed |= set_ann(node,'fraud',0.8,1.0,'R5: suspicious → fraud')

    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'transaction' and meets(node,'fraud',0.8):
            for recv in G.successors(node):
                if G.edges[node,recv].get('relation') == 'receives':
                    changed |= set_ann(recv,'fraud_risk',0.6,0.85,
                                       f'R6: fraud({node})')
    return changed

with st.spinner("Running annotated logic reasoning..."):
    for t in range(10):
        if not run_rules(G):
            break

#  Results 
flagged = []
for idx, row in df.iterrows():
    tx_id    = row['tx_id']
    fraud_ann = get_ann(tx_id, 'fraud')
    is_flagged = fraud_ann[0] >= 0.8

    if is_flagged or show_legit:
        fired = rule_trace.get(tx_id, [])
        ann   = {k: v for k,v in annotations[tx_id].items() if v[0] > 0}
        entry = {
            'tx_id'          : tx_id,
            'type'           : row['type'],
            'amount'         : row['amount'],
            'nameOrig'       : row['nameOrig'],
            'nameDest'       : row['nameDest'],
            'flagged'        : '🚨 Fraud' if is_flagged else '✅ Legit',
            'fraud_lb'       : fraud_ann[0],
            'rules_fired'    : len(fired),
            'rule_trace'     : fired,
            'all_annotations': ann,
        }
        if has_labels:
            entry['ground_truth'] = '🚨 Fraud' if row['isFraud'] else '✅ Legit'
        flagged.append(entry)

df_results = pd.DataFrame(flagged)
n_flagged  = (df_results['flagged'] == '🚨 Fraud').sum() if len(df_results) > 0 else 0

# Metrics show
st.markdown("---")
st.subheader("Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions analyzed", f"{len(df):,}")
col2.metric("Flagged as fraud",       f"{n_flagged:,}")
col3.metric("Large amount threshold", f"${LARGE_AMOUNT_THRESHOLD:,.0f}")
col4.metric("Converged at timestep",  f"{t+1}")

#Results tabel
st.markdown("---")
st.subheader("Flagged transactions")

if len(df_results) == 0:
    st.warning("No transactions flagged. Try lowering the thresholds in the sidebar.")
else:
    display_cols = ['tx_id','type','amount','nameOrig','nameDest','flagged','rules_fired']
    if has_labels:
        display_cols.append('ground_truth')
    st.dataframe(df_results[display_cols], use_container_width=True, height=300)

#Rule trace [Dora the] explorer
st.markdown("---")
st.subheader("Rule trace explorer")
st.caption("Select any transaction to see exactly why it was flagged — full audit trail.")

flagged_ids = df_results[df_results['flagged']=='🚨 Fraud']['tx_id'].tolist() if len(df_results) > 0 else []

if not flagged_ids:
    st.info("No flagged transactions to inspect.")
else:
    selected = st.selectbox("Select a flagged transaction", flagged_ids)
    row = df_results[df_results['tx_id'] == selected].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Transaction details**")
        st.write(f"Type: `{row['type']}`")
        st.write(f"Amount: `${row['amount']:,.2f}`")
        st.write(f"From: `{row['nameOrig']}`")
        st.write(f"To: `{row['nameDest']}`")
        if has_labels:
            st.write(f"Ground truth: {row['ground_truth']}")

    with col2:
        st.markdown("**Annotation intervals**")
        for pred, bounds in row['all_annotations'].items():
            bar_val = bounds[0]
            st.write(f"`{pred:<22}` [{bounds[0]:.2f}, {bounds[1]:.2f}]")
            st.progress(bar_val)

    st.markdown("**Rules that fired (in order)**")
    for i, rule in enumerate(row['rule_trace']):

        st.write(f"{i+1}. {rule}")
