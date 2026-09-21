import pandas as pd
import networkx as nx

edges = pd.read_csv("protein_network_edges.csv") 
G = nx.from_pandas_edgelist(edges, source="Source", target="Target")


# Basic information
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
# Remove self-loops (proteins "interacting with themselves" — clutters visualization)
G.remove_edges_from(list(nx.selfloop_edges(G)))

# Work on the largest connected component LCC only
largest = max(nx.connected_components(G), key=len)
G = G.subgraph(largest).copy()

print("After cleanup -> ")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
#community detection by Louvain
communities = nx.community.louvain_communities(G, seed=42)#random seed
modularity = nx.community.modularity(G, communities)#modularity 0-1 score, >0.3 meaningful

print("Number of communities:", len(communities))
print("Modularity:", round(modularity, 4))
#only big communities, small ones aree grouped together
sized = sorted(communities, key=len, reverse=True)
main_communities = [c for c in sized if len(c) >= 15]   # adjustable max, can change as we like
other_nodes = set()
for c in sized:
    if len(c) < 15: #anything under 15 nodes grouped togeter
        other_nodes.update(c)

print("Main communities:", len(main_communities))
print("Small communities merged into 'other':", len(other_nodes), "nodes")
#Separation Process
import numpy as np

n_clusters = len(main_communities) + (1 if other_nodes else 0)
angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)

# Scale spacing, so bigger communities have room without overlapping neighbors
local_scale_factor = 0.35
max_local_radius = local_scale_factor * np.sqrt(max(len(c) for c in main_communities))
overall_radius = n_clusters * (2 * max_local_radius) / (2 * np.pi) * 1.4
#centeroid-home of each community
centroids = {}
for i in range(len(main_communities)):
    centroids[i] = (overall_radius*np.cos(angles[i]), overall_radius*np.sin(angles[i]))
if other_nodes:
    centroids['other'] = (overall_radius*np.cos(angles[-1]), overall_radius*np.sin(angles[-1]))

pos = {}
comm_id_map = {}

for i, c in enumerate(main_communities):
    sub = G.subgraph(c)
    local_scale = local_scale_factor * np.sqrt(len(c))
    local_pos = nx.spring_layout(sub, seed=42, k=1.2/np.sqrt(len(c)), iterations=50)#computes a separate layout for each community 
    cx, cy = centroids[i]
    for node, (x, y) in local_pos.items():
        pos[node] = (cx + x*local_scale, cy + y*local_scale)#shift of small local layout so it's centered on the communities's assigned spot
        comm_id_map[node] = i

if other_nodes:
    sub = G.subgraph(other_nodes)
    local_scale = local_scale_factor * np.sqrt(len(other_nodes))
    local_pos = nx.spring_layout(sub, seed=42, k=1.2/np.sqrt(len(other_nodes)), iterations=50)
    cx, cy = centroids['other']
    for node, (x, y) in local_pos.items():
        pos[node] = (cx + x*local_scale, cy + y*local_scale)
        comm_id_map[node] = 'other'
#Drawing the Graph
import matplotlib.pyplot as plt
import random

n_main = len(main_communities)
cmap = plt.colormaps.get_cmap('tab20')
color_lookup = {i: cmap(i / max(n_main-1,1)) for i in range(n_main)}
color_lookup['other'] = (0.7, 0.7, 0.7, 1.0)

nodes_in_pos = [n for n in G.nodes() if n in pos]
node_colors = [color_lookup[comm_id_map[n]] for n in nodes_in_pos]

# Split edges: within-community vs. between-community
intra_edges, inter_edges = [], []
for u, v in G.edges():
    if u not in comm_id_map or v not in comm_id_map:
        continue
    (intra_edges if comm_id_map[u] == comm_id_map[v] else inter_edges).append((u, v))

random.seed(42)
inter_sample = random.sample(inter_edges, min(13329, len(inter_edges)))  #Sample of 1500 nodes in bvetween communities

fig, ax = plt.subplots(figsize=(16, 16))
nx.draw_networkx_edges(G, pos, edgelist=inter_sample, ax=ax, alpha=0.06, width=0.4, edge_color='gray')
nx.draw_networkx_edges(G, pos, edgelist=intra_edges, ax=ax, alpha=0.08, width=0.3, edge_color='gray')
nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_pos, ax=ax, node_size=8, node_color=node_colors, linewidths=0)

ax.set_title(f"Community-Separated Layout ({n_main} major communities)")
ax.axis('off')
plt.tight_layout()
plt.savefig("community_separated_layout.png", dpi=150)
plt.show()