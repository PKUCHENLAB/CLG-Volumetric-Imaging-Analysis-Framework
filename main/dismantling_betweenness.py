#Zebrafishproject– betweenness centrality version

from __future__ import annotations
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt


_GPU_SUCCESS = False
_GPU_ERROR_MSG = None

try:
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cudf
        import cugraph
    _GPU_SUCCESS = True
    print("GPU backend detected, attempting to initialize...")
except ImportError as e:
    _GPU_ERROR_MSG = str(e)
    print("GPU packages not available, switching to graph-tool on CPU")

if _GPU_SUCCESS:
    try:
        
        smoke_df = cudf.DataFrame({"src": [0, 1], "dst": [1, 2]})
        g_test = cugraph.Graph()
        g_test.from_cudf_edgelist(smoke_df, source="src", destination="dst")
        del smoke_df, g_test
    except Exception as e:
        _GPU_SUCCESS = False
        _GPU_ERROR_MSG = str(e)
        print("GPU backend initialization failed:", _GPU_ERROR_MSG)
        print("Continuing with graph-tool on CPU")

import graph_tool.all as gt

def _lcc_cugraph(g, removed: set[int], orig_size: int) -> float:
    mask = ~(
        g.edgelist.edgelist_df['src'].isin(removed) |
        g.edgelist.edgelist_df['dst'].isin(removed)
    )
    g_sub = cugraph.Graph()
    g_sub.from_cudf_edgelist(
        g.edgelist.edgelist_df[mask],
        source='src', destination='dst', store_transposed=True
    )
    wcc = cugraph.weakly_connected_components(g_sub)
    counts = wcc['labels'].value_counts()
    lcc = counts.max() if len(counts) else 0
    return (lcc / orig_size) * 100


def _betweenness_cugraph(g):
    bc = cugraph.betweenness_centrality(g, normalized=True)
    return bc.sort_values(
        by="betweenness_centrality", ascending=False
    )['vertex'].to_arrow().to_pylist()


def _lcc_graph_tool(g_gt: gt.Graph, removed: set[int], orig_size: int) -> float:
    g = gt.Graph(g_gt)
    vfilt = g.new_vertex_property("bool")
    for v in g.vertices():
        vfilt[v] = int(v) not in removed
    g.set_vertex_filter(vfilt)
    if g.num_vertices() == 0:
        return 0.0
    comp, _ = gt.label_components(g, directed=g.is_directed())
    lcc_size = max(np.bincount(comp.a))
    return (lcc_size / orig_size) * 100


def _betweenness_graph_tool(g_gt: gt.Graph):
    vp, _ = gt.betweenness(g_gt)
    pairs = [(int(v), vp[v]) for v in g_gt.vertices()]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [v for v, _ in pairs]


def main():
    start_total = time.time()

    graph_file = "/usr/bin/env/yourdataset/yourdata.gt"
    output_dir = "/usr/bin/env/yourdataset/out"
    os.makedirs(output_dir, exist_ok=True)

    g_gt = gt.load_graph(graph_file)
    max_vid = int(g_gt.get_vertices().max())
    print("Graph-tool vertices:", g_gt.num_vertices())
    print("Graph-tool edges   :", g_gt.num_edges())

    if _GPU_SUCCESS:
        
        import cudf
        all_nodes = np.arange(max_vid + 1, dtype=np.int32)
        isolate_df = cudf.DataFrame({
            "src": all_nodes,
            "dst": np.full_like(all_nodes, -1, dtype=np.int32)
        })
        edges = [(int(e.source()), int(e.target())) for e in g_gt.edges()]
        if edges:
            real_df = cudf.DataFrame({"src": [u for u, v in edges],
                                      "dst": [v for u, v in edges]})
            full_df = cudf.concat([real_df, isolate_df])
        else:
            full_df = isolate_df
        full_df = full_df[full_df["dst"] != -1]

        g_cu = cugraph.Graph()
        g_cu.from_cudf_edgelist(
            full_df, source="src", destination="dst", store_transposed=True
        )

        wcc = cugraph.weakly_connected_components(g_cu)
        original_lcc_size = wcc['labels'].value_counts().max()

        start = time.time()
        sorted_nodes = _betweenness_cugraph(g_cu)
        print(f"Betweenness centrality time: {time.time() - start:.2f} s")

        lcc_func = lambda rem: _lcc_cugraph(g_cu, rem, original_lcc_size)

    else:
        
        comp, _ = gt.label_components(g_gt, directed=g_gt.is_directed())
        original_lcc_size = max(np.bincount(comp.a))

        start = time.time()
        sorted_nodes = _betweenness_graph_tool(g_gt)
        print(f"Betweenness centrality time: {time.time() - start:.2f} s")

        lcc_func = lambda rem: _lcc_graph_tool(g_gt, rem, original_lcc_size)

    print("Initial largest component size:", original_lcc_size)


    target_lcc = 20.0
    max_nodes_to_remove = 100000
    removed_nodes = []
    lcc_values = [100.0]
    nodes_removed_list = [0]

    for node in sorted_nodes:
        if lcc_values[-1] <= target_lcc or len(removed_nodes) >= max_nodes_to_remove:
            break

        removed_nodes.append(int(node))
        current_lcc = lcc_func(set(removed_nodes))
        lcc_values.append(current_lcc)
        nodes_removed_list.append(len(removed_nodes))

        print(f"Removed {len(removed_nodes):5d} nodes, LCC = {current_lcc:6.2f}%")


    plt.figure(figsize=(12, 7))
    plt.plot(nodes_removed_list, lcc_values, marker='o', color='tab:blue')
    plt.axhline(y=100.0, color='green', linestyle='--')
    plt.axhline(y=target_lcc, color='red', linestyle='--')
    plt.xlabel('Nodes Removed')
    plt.ylabel('LCC (%)')
    plt.title('Mouse Dismantling (Betweenness, Target 20%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lcc_curve_20pct.png"))

    np.save(os.path.join(output_dir, "removed_nodes_20pct.npy"),
            np.array(removed_nodes, dtype=np.int64))
    np.save(os.path.join(output_dir, "lcc_values_20pct.npy"),
            np.array(lcc_values, dtype=np.float32))
    np.save(os.path.join(output_dir, "nodes_removed_list_20pct.npy"),
            np.array(nodes_removed_list, dtype=np.int64))

    import pandas as pd
    pd.DataFrame({"node_id": removed_nodes}).to_csv(
        os.path.join(output_dir, "node_decomposition_20pct.csv"), index=False)

    print(f"Finished. Removed {len(removed_nodes)} nodes in total, "
          f"elapsed {time.time() - start_total:.2f} s")

if __name__ == "__main__":
    main()