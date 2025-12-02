# Zebrafishproject – zebragdm version

import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        import cudf
        import cugraph
        import cupy
        USE_GPU = True
except Exception:
    pass

if USE_GPU:
    BACKEND = "gpu"
else:
    import networkx as nx
    BACKEND = "cpu"

print(f"[Backend] Using {BACKEND.upper()} backend")

class GraphBackend:

    def __init__(self, edges, n_nodes):
        self.n_nodes = n_nodes
        self._edges = edges
        if USE_GPU:
            df = cudf.DataFrame({"src": [u for u, v in edges],
                                 "dst": [v for u, v in edges]})
            self._G = cugraph.Graph()
            self._G.from_cudf_edgelist(df, source="src", destination="dst")
        else:
            self._G = nx.Graph()
            self._G.add_nodes_from(range(n_nodes))
            self._G.add_edges_from(edges)

    def number_of_nodes(self):
        return self.n_nodes

    def number_of_edges(self):
        return len(self._edges)

    def largest_cc_size(self):
        if USE_GPU:
            wcc = cugraph.weakly_connected_components(self._G)
            return int(wcc["labels"].value_counts().max())
        else:
            if self._G.number_of_nodes() == 0:
                return 0
            max_cc = max(nx.connected_components(self._G), key=len, default=set())
            return len(max_cc)

    def remove_nodes(self, nodes):
        if USE_GPU:
            df = cudf.DataFrame({"src": [u for u, v in self._edges],
                                 "dst": [v for u, v in self._edges]})
            mask = ~(df["src"].isin(nodes) | df["dst"].isin(nodes))
            df_new = df[mask]
            edges_new = list(zip(df_new["src"].to_pandas().tolist(),
                                 df_new["dst"].to_pandas().tolist()))
        else:
            G_new = self._G.copy()
            G_new.remove_nodes_from(nodes)
            edges_new = list(G_new.edges)
        return GraphBackend(edges_new, self.n_nodes - len(nodes))

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PyTorch] Using device: {device}")

import torch.nn.functional as F
from torch_geometric.nn import GATConv
import graph_tool.all as gt

class GAT_Model(torch.nn.Module):
    def __init__(self, input_dim, conv_layers, heads, concat,
                 negative_slope, dropout, bias):
        super().__init__()
        self.gat_layers = torch.nn.ModuleList()
        for i, (dim, h, c, ns, do, bi) in enumerate(
                zip(conv_layers, heads, concat, negative_slope, dropout, bias)):
            in_ch = conv_layers[i - 1] * heads[i - 1] if i else input_dim
            self.gat_layers.append(GATConv(in_ch, dim, heads=h, concat=c,
                                           negative_slope=ns, dropout=do, bias=bi))
        self.fc = torch.nn.Linear(conv_layers[-1] * heads[-1], 1)

    def forward(self, x, edge_index):
        for layer in self.gat_layers:
            x = F.elu(layer(x, edge_index))
        x = self.fc(x)
        return torch.sigmoid(x)

def calculate_features(g_gt):

    n = g_gt.num_vertices()
    feats = np.zeros((n, 8), dtype=np.float32)

    degree = g_gt.get_out_degrees(g_gt.get_vertices())
    max_d = max(degree.max(), 1)
    feats[:, 0] = degree / max_d

    mean_d = degree.mean()
    mean_d = max(mean_d, 1e-8)
    feats[:, 1] = ((degree - mean_d) ** 2) / mean_d

    try:
        _, ev = gt.eigenvector(g_gt)
        ev_arr = ev.get_array()
        if ev_arr.dtype.kind in 'biufc':
            feats[:, 2] = ev_arr
        else:
            feats[:, 2] = 0.0
    except Exception:
        feats[:, 2] = 0.0


    cc = gt.local_clustering(g_gt)
    cc_arr = cc.get_array()
    feats[:, 3] = cc_arr
    mean_cc = max(cc_arr.mean(), 1e-8)
    feats[:, 4] = ((cc_arr - mean_cc) ** 2) / mean_cc


    pr = gt.pagerank(g_gt)
    feats[:, 5] = pr.get_array()


    bc, _ = gt.betweenness(g_gt)
    bc_arr = bc.get_array()
    feats[:, 6] = bc_arr if bc_arr.dtype.kind in 'biufc' else 0.0


    kcore = gt.kcore_decomposition(g_gt)
    k_arr = kcore.get_array()
    max_k = max(k_arr.max(), 1)
    feats[:, 7] = k_arr / max_k


    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats

def normalize_features(features):
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    rng = max_vals - min_vals
    rng[rng == 0] = 1
    return (features - min_vals) / rng


def load_graph(file_path):
    edges = []
    if file_path.endswith(".h5"):
        with h5py.File(file_path, "r") as f:
            adj = f["adjacency"][:]
        if USE_GPU:
            G = cugraph.Graph()
            G.from_cupy_sparse_matrix(cupy.sparse.csr_matrix(adj))
            df = G.view_edge_list()
            edges = list(zip(df["src"].to_pandas().tolist(),
                             df["dst"].to_pandas().tolist()))
            n = adj.shape[0]
        else:
            G = nx.from_numpy_array(adj)
            edges = list(G.edges)
            n = G.number_of_nodes()
    elif file_path.endswith(".gt"):
        g_gt = gt.load_graph(file_path)
        edges = [(int(e.source()), int(e.target())) for e in g_gt.edges()]
        n = g_gt.num_vertices()
    else:
        raise ValueError("Only .h5 or .gt supported")
    return GraphBackend(edges, n)


def main():
    graph_file = "/usr/bin/env/yourdataset/yourdata.gt"
    output_dir = "/usr/bin/env/yourdataset/out"
    os.makedirs(output_dir, exist_ok=True)

    print("[Step 1/4] Loading graph...")
    g = load_graph(graph_file)
    print(f"  -> Loaded: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    original_lcc = g.largest_cc_size()
    print(f"  -> Original LCC size: {original_lcc}")


    g_gt = gt.Graph(directed=False)
    g_gt.add_vertex(g.number_of_nodes())
    for u, v in g._edges:
        g_gt.add_edge(u, v)


    model = GAT_Model(
        input_dim=8,
        conv_layers=[64, 32],
        heads=[4, 4],
        concat=[True, True],
        negative_slope=[0.2, 0.2],
        dropout=[0.3, 0.3],
        bias=[True, True]
    ).to(device)

    model_path = "/usr/bin/env/yourdataset/gdm_trained_model.pth"
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("  -> GAT model loaded and ready")


    target_lcc = 18.0
    max_remove = 100_000
    removed = []
    current_lcc_perc = 100.0
    lcc_list, n_list = [100.0], [0]

    iteration = 0
    while current_lcc_perc > target_lcc and len(removed) < max_remove:
        if g.number_of_nodes() < 2:
            break

        iteration += 1

        features = calculate_features(g_gt)
        features = normalize_features(features)
        x = torch.tensor(features, dtype=torch.float32, device=device)


        n = g.number_of_nodes()
        if n < 2:
            break
        ii, jj = torch.tril_indices(n, n, -1, device=device)
        edge_index = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0)

        with torch.no_grad():
            scores = model(x, edge_index).squeeze().cpu().numpy()


        bet = gt.betweenness(g_gt)[0].get_array()
        clo = gt.closeness(g_gt).get_array()
        deg = np.array([g_gt.vertex(i).out_degree() for i in range(n)])


        def norm(a):
            mx = a.max() if a.max() != 0 else 1
            return a / mx
        combined = (a * norm(scores) +
                    b * norm(bet) +
                    c * norm(clo) +
                    d * norm(deg))

        next_node = int(combined.argmax())
        removed.append(next_node)


        g = g.remove_nodes([next_node])
        g_gt.remove_vertex(next_node, fast=True)

        current_lcc_perc = (g.largest_cc_size() / original_lcc) * 100
        lcc_list.append(current_lcc_perc)
        n_list.append(len(removed))
        print(f"  -> Iter {iteration}: removed node {next_node}, "
              f"LCC={current_lcc_perc:.2f}%")

    print("[Step 2/4] Dismantling finished")
    print(f"  -> Total nodes removed: {len(removed)}")


    plt.figure(figsize=(12, 7))
    plt.plot(n_list, lcc_list, marker='o', color='blue')
    plt.axhline(y=100, color='green', linestyle='--')
    plt.axhline(y=target_lcc, color='red', linestyle='--')
    plt.xlabel("Nodes Removed")
    plt.ylabel("LCC (%)")
    plt.title("Zebragdm LCC Curve")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "lcc_curve_universal.png"))

    np.save(os.path.join(output_dir, "removed_nodes.npy"), removed)
    np.save(os.path.join(output_dir, "lcc_values.npy"), lcc_list)
    np.save(os.path.join(output_dir, "nodes_removed_list.npy"), n_list)
    print(f"  -> Results saved to {output_dir}")

if __name__ == "__main__":
    main()