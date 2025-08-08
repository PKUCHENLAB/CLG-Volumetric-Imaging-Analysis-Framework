import os
import time
import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt
import cudf
import cugraph
import pandas as pd

# --------------------------------------------------
# 移除节点后返回剩余最大连通分量百分比
# --------------------------------------------------
def calculate_lcc_after_removal(g_cugraph, removed_nodes, original_lcc_size):
    mask = ~(
        g_cugraph.edgelist.edgelist_df['src'].isin(removed_nodes) |
        g_cugraph.edgelist.edgelist_df['dst'].isin(removed_nodes)
    )
    g_sub = cugraph.Graph()
    g_sub.from_cudf_edgelist(
        g_cugraph.edgelist.edgelist_df[mask],
        source="src",
        destination="dst",
        store_transposed=True
    )
    wcc_df = cugraph.weakly_connected_components(g_sub)
    counts = wcc_df['labels'].value_counts()
    lcc_size = counts.max() if len(counts) else 0
    return (lcc_size / original_lcc_size) * 100

# --------------------------------------------------
# 主程序
# --------------------------------------------------
def main():
    start_total = time.time()

    graph_file = "/gpfs/share/home/2306391536/zhaojunjie/zebramouse/data/tout/trace_network_08.gt"
    output_dir = "/gpfs/share/home/2306391536/zhaojunjie/zebramouse/data/tout/lccout"
    os.makedirs(output_dir, exist_ok=True)

    # 1. graph-tool 读取
    g_gt = gt.load_graph(graph_file)
    max_vid = int(g_gt.get_vertices().max())
    edges = [(int(e.source()), int(e.target())) for e in g_gt.edges()]
    print("graph-tool 节点数:", g_gt.num_vertices())
    print("graph-tool 边数  :", g_gt.num_edges())

    # 2. 构造 cuGraph：一次性加入所有节点（含孤立节点）+ 真实边
    #    方法：把所有节点 0..max_vid 做成 DataFrame，再加真实边
    all_nodes = np.arange(max_vid + 1, dtype=np.int32)
    isolate_df = cudf.DataFrame({
        "src": all_nodes,
        "dst": np.full_like(all_nodes, -1, dtype=np.int32)  # 占位，稍后过滤
    })
    # 真实边
    if edges:
        real_df = cudf.DataFrame({"src": [u for u, v in edges],
                                  "dst": [v for u, v in edges]})
        full_df = cudf.concat([real_df, isolate_df])
    else:
        full_df = isolate_df

    # 过滤占位边 (-1)
    full_df = full_df[full_df["dst"] != -1]

    # 建图
    g_cugraph = cugraph.Graph()
    g_cugraph.from_cudf_edgelist(full_df, source="src", destination="dst", store_transposed=True)

    print("cuGraph 节点数   :", g_cugraph.number_of_vertices())
    print("cuGraph 边数     :", g_cugraph.number_of_edges())

    # 3. 原始最大连通分量
    wcc_df = cugraph.weakly_connected_components(g_cugraph)
    original_lcc_size = wcc_df['labels'].value_counts().max()
    print("原始最大连通分量 :", original_lcc_size)

    # 4. 介数中心性
    start = time.time()
    bc = cugraph.betweenness_centrality(g_cugraph, normalized=True)
    bc = bc.sort_values(by="betweenness_centrality", ascending=False)
    sorted_nodes = bc['vertex'].to_arrow().to_pylist()
    print(f"计算介数中心性耗时: {time.time() - start:.2f} s")

    # 5. 拆解循环
    target_lcc = 20.0
    max_nodes_to_remove = 100_000
    removed_nodes = []
    lcc_values = [100.0]
    nodes_removed_list = [0]

    for node in sorted_nodes:
        if lcc_values[-1] <= target_lcc or len(removed_nodes) >= max_nodes_to_remove:
            break

        removed_nodes.append(int(node))
        current_lcc = calculate_lcc_after_removal(g_cugraph, removed_nodes, original_lcc_size)
        lcc_values.append(current_lcc)
        nodes_removed_list.append(len(removed_nodes))

        print(f"已移除 {len(removed_nodes):5d} 节点，LCC = {current_lcc:6.2f}%")

    # 6. 可视化 & 保存
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

    pd.DataFrame({"node_id": removed_nodes}).to_csv(
        os.path.join(output_dir, "node_decomposition_20pct.csv"), index=False)

    print(f"结束。共移除 {len(removed_nodes)} 节点，总耗时 {time.time() - start_total:.2f} s")

if __name__ == "__main__":
    main()