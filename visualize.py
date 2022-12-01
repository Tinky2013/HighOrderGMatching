import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

A = np.array(pd.read_csv('data/synthetic_dt/A_0_3_0.1_100_N/net_11.csv'))	# matrix (num_nodes, num_nodes)
A = A - sp.dia_matrix((A.diagonal()[np.newaxis, :], [0]), shape=A.shape)
A = sp.coo_matrix(A)
coords = [(A.row[i], A.col[i]) for i in range(len(A.row))]

G=nx.Graph()
G.add_edges_from(coords)
nx.draw(G, with_labels=False,
        node_size=5,
        width=0.5,
        edge_color='purple',
        node_color='black')
plt.show()