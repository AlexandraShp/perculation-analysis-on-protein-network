from pathlib import Path
import sys

import pandas as pd
import numpy as np
import networkx as nx

# ---------------------------------------------------------
# Locate project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from network import load_baseline_network

OUTPUT_DIR = PROJECT_ROOT / "visualisation" / "Gephi$"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Output directory:", OUTPUT_DIR.resolve())

# ---------------------------------------------------------
# Load baseline network + communities
# ---------------------------------------------------------
print("Loading baseline network G0...")
G0 = load_baseline_network()
print("G0:", G0.number_of_nodes(), "nodes,", G0.number_of_edges(), "edges")

communities = nx.community.louvain_communities(G0, seed=42)
node_to_comm = {}
for i, c in enumerate(communities):
    for n in c:
        node_to_comm[n] = i
print("Communities:", len(communities))

# ---------------------------------------------------------
# Function: build dynamic GEXF from a saved removal history
# ---------------------------------------------------------
def build_dynamic_gexf_from_history(G0, removal_history_path, node_to_comm, out_path):
    hist = pd.read_csv(removal_history_path)
    removal_step = dict(zip(hist["node"], hist["removal_step"]))
    max_step = int(hist["removal_step"].max())

    G = G0.copy()
    G.graph['mode'] = 'dynamic'
    G.graph['start'] = 0
    G.graph['end'] = max_step

    for n in G.nodes():
        end_time = removal_step.get(n, max_step)
        G.nodes[n]['start'] = 0
        G.nodes[n]['end'] = int(end_time)
        G.nodes[n]['community'] = int(node_to_comm.get(n, -1))

    for u, v in G.edges():
        end_time = min(G.nodes[u]['end'], G.nodes[v]['end'])
        G.edges[u, v]['start'] = 0
        G.edges[u, v]['end'] = int(end_time)

    nx.write_gexf(G, out_path, version="1.2draft")
    print("Saved:", out_path)

# ---------------------------------------------------------
# Function: build dynamic GEXF for random removal (regenerated, same seed as your script)
# ---------------------------------------------------------
def build_random_dynamic_gexf(G0, node_to_comm, out_path, seed=42, trial=1):
    rng = np.random.default_rng(seed + trial)
    nodes_list = list(G0.nodes())
    removal_order = rng.permutation(nodes_list)
    removal_step = {node: i + 1 for i, node in enumerate(removal_order)}

    G = G0.copy()
    G.graph['mode'] = 'dynamic'
    G.graph['start'] = 0
    G.graph['end'] = len(nodes_list)

    for n in G.nodes():
        G.nodes[n]['start'] = 0
        G.nodes[n]['end'] = int(removal_step[n])
        G.nodes[n]['community'] = int(node_to_comm.get(n, -1))

    for u, v in G.edges():
        end_time = min(G.nodes[u]['end'], G.nodes[v]['end'])
        G.edges[u, v]['start'] = 0
        G.edges[u, v]['end'] = int(end_time)

    nx.write_gexf(G, out_path, version="1.2draft")
    print("Saved:", out_path)

# ---------------------------------------------------------
# ACTUALLY RUN both builds  <-- this is the part that was missing
# ---------------------------------------------------------
history_file = (
    PROJECT_ROOT / "results" / "targeted" / "adaptive_targeted"
    / "adaptive_targeted_removal_history.csv"
)

build_dynamic_gexf_from_history(
    G0,
    history_file,
    node_to_comm,
    OUTPUT_DIR / "adaptive_targeted_dynamic.gexf"
)

build_random_dynamic_gexf(
    G0, node_to_comm,
    OUTPUT_DIR / "random_dynamic.gexf"
)

print("\nAll done. Files in:", OUTPUT_DIR.resolve())