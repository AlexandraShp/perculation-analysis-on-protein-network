import pandas as pd
import networkx as nx

# Load the dataset
df = pd.read_csv(
    "pcbi.1004120.s003 (10).tsv",
    sep="\t",
    comment="#",
    header=None,
    names=[
        "gene_ID_1",
        "gene_ID_2",
        "gene_symbol_1",
        "gene_symbol_2",
        "sources"
    ]
)

# Show the data
print(df.head())

# Keep only the columns needed for the network
edges = df[["gene_ID_1", "gene_ID_2"]]

# Build the graph
G = nx.from_pandas_edgelist(
    edges,
    source="gene_ID_1",
    target="gene_ID_2"
)

# Basic information
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())