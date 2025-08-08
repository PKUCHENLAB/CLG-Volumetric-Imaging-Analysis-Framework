#zebrafish-project
#!/usr/bin/env python
# corr_analysis.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  

file_path = '/usr/bin/env/yourdataset/yourdate.h5'
with h5py.File(file_path, 'r') as f:
    data = f['trace'][:]        
print('Loaded data shape:', data.shape)

corr_mat = np.corrcoef(data)     

mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
corr_vals = corr_mat[mask]      

print('Total correlation pairs:', corr_vals.size)

percentiles = [80, 85, 90, 95]
thresholds = {p: np.percentile(corr_vals, p) for p in percentiles}

print('\nCorrelation thresholds:')
for p in percentiles:
    print(f'  P{p:2d} = {thresholds[p]:.4f}')

plt.figure(figsize=(8, 5))
plt.hist(corr_vals, bins=200, density=True, color='steelblue', alpha=0.7)

colors = sns.color_palette('colorblind', n_colors=len(percentiles))

for p, c in zip(percentiles, colors):
    plt.axvline(thresholds[p],
                color=c,
                linestyle='--',
                linewidth=1.5,
                alpha=0.75,
                label=f'P{p} ({thresholds[p]:.3f})')

plt.xlabel('Pearson correlation')
plt.ylabel('Density')
plt.title('Distribution of pairwise correlations (upper triangle)')
plt.legend()
plt.tight_layout()

save_path = '/usr/bin/env/yourdataset/corr_distribution.png'
plt.savefig(save_path, dpi=300)
print('\nFigure saved to:', save_path)

plt.show()