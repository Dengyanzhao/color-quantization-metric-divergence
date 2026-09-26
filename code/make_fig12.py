# -*- coding: utf-8 -*-
"""重绘 fig12：三数据集 CIEDE2000 对比（Kodak / BSDS300 / DIV2K，K=64）。
全部使用完整版 CIEDE2000 重跑数据。"""

# --- path setup added for public release ---
# The original scripts referenced a local absolute path that contained
# the author's name; it is now resolved relative to this file instead.
# Override with the PROJECT_ROOT env var if needed.
#
# Raw image datasets are NOT bundled (they are public and large); see the
# dataset URLs in README.md. Set DATASETS_DIR to your local copy:
#   $env:DATASETS_DIR = 'C:/my/datasets'      # PowerShell
#   export DATASETS_DIR=$HOME/datasets        # bash
import os as _os

_ROOT = _os.environ.get(
    'PROJECT_ROOT',
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

_DATASETS = _os.environ.get('DATASETS_DIR', _os.path.join(_ROOT, 'datasets'))

# Root of this checkout, where data/ and figures/ live.
ROOT = _ROOT

# Directory of this script (so sibling modules can be imported by name).
_CODE = _os.path.dirname(_os.path.abspath(__file__))
# --- end path setup ---

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
                     'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})
D = f'{_ROOT}/data'
FIGS = [f'{_ROOT}/figures', f'{_ROOT}/figures']

# Kodak: CE 来自 ce_results.csv，kmeans/pw 来自 metric_full_matrix.csv
kce = pd.read_csv(f'{D}/ce_results.csv')
kce64 = kce[kce.n_colors == 64]
mfm = pd.read_csv(f'{D}/metric_full_matrix.csv')
mfm64 = mfm[mfm.n_colors == 64]
kodak = {
    'K-means': mfm64[mfm64.method == 'kmeans_rgb'].dE00.mean(),
    'PW-Lab': mfm64[mfm64.method == 'pw_lab'].dE00.mean(),
    'CE-KMeans': kce64.dE00.mean(),
}
# BSDS / DIV2K
def load3(path, k=64):
    df = pd.read_csv(path)
    if 'n_colors' in df.columns:
        df = df[df.n_colors == k]
    g = df.groupby('method').dE00.mean()
    return {'K-means': g['kmeans_rgb'], 'PW-Lab': g['pw_lab'], 'CE-KMeans': g['ce_kmeans']}

bsds = load3(f'{D}/bsd100_methods.csv')
div2k = load3(f'{D}/div2k_methods.csv')

print('Kodak:', {k: round(v, 3) for k, v in kodak.items()})
print('BSDS300:', {k: round(v, 3) for k, v in bsds.items()})
print('DIV2K:', {k: round(v, 3) for k, v in div2k.items()})

datasets = ['Kodak (24)', 'BSDS300 (100)', 'DIV2K (100)']
methods = ['K-means', 'PW-Lab', 'CE-KMeans']
colors = ['#c44e52', '#4c72b0', '#55a868']
data = [kodak, bsds, div2k]

fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(datasets))
w = 0.26
for i, (m, c) in enumerate(zip(methods, colors)):
    vals = [d[m] for d in data]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=m, color=c)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.2f}',
                ha='center', fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel('Mean CIEDE2000 ($\\Delta E_{00}$)')
ax.set_title('CIEDE2000 comparison across three datasets (K = 64)')
ax.legend(frameon=False, ncol=3, fontsize=8)
ax.set_ylim(0, max(max(d.values()) for d in data) * 1.2)
plt.tight_layout()
for fdir in FIGS:
    plt.savefig(os.path.join(fdir, 'fig12_three_datasets.pdf'))
    plt.savefig(os.path.join(fdir, 'fig12_three_datasets.png'), dpi=150)
plt.close()
print('fig12 已重绘并同步到两处')
