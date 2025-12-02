#ZebrafishProject-build_network

import os
import numpy as np
import h5py
import graph_tool.all as gt

src_file = '/usr/bin/env/yourdataset/yourdata.h5'
out_dir  = '/usr/bin/env/yourdataset/out'
os.makedirs(out_dir, exist_ok=True)

with h5py.File(src_file, 'r') as f:
    data = f['trace'][:]
print('Loaded data shape:', data.shape)

corr_mat = np.abs(np.corrcoef(data))
print('Correlation matrix shape:', corr_mat.shape)

threshold = 0.8
print('Fixed threshold (abs):', threshold)

n_nodes = corr_mat.shape[0]
g = gt.Graph(directed=False)
g.add_vertex(n_nodes)

edges, weights = [], []
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        w = corr_mat[i, j]
        if w >= threshold:
            edges.append((i, j))
            weights.append(w)

g.add_edge_list(edges)
ew = g.new_edge_property("double")
ew.a = weights
g.edge_properties['weight'] = ew

print('Graph built: nodes =', g.num_vertices(), 'edges =', g.num_edges())

gt_path = os.path.join(out_dir, 'trace_network_08.gt')
g.save(gt_path)
print('Graph saved to:', gt_path)

pos = gt.sfdp_layout(g)
deg = g.degree_property_map("total")
deg.a = np.log2(deg.a + 1)

fig_path = os.path.join(out_dir, 'trace_network_08.png')
gt.graph_draw(
    g,
    pos=pos,
    vertex_size=gt.prop_to_size(deg, mi=2, ma=10),
    vertex_fill_color=deg,
    edge_pen_width=0.5,
    output=fig_path,
    output_size=(2000, 2000)
)
print('Visualization saved to:', fig_path)
