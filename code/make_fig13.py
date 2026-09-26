# -*- coding: utf-8 -*-
"""重绘 fig13：K=64 三方法四指标归一化对比（含新 LPIPS 值）。
数据源：metric_full_matrix.csv（psnr/ssim/dE00/lpips 合并宽表）。"""

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

plt.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight', 'font.size': 9})
D = f'{_ROOT}/data'
FIG = f'{_ROOT}/figures'

df = pd.read_csv(os.path.join(D, 'metric_full_matrix.csv'))
k64 = df[df.n_colors == 64]
methods = ['kmeans_rgb', 'pw_lab', 'ce_kmeans']
labels = ['K-means', 'PW-Lab', 'CE-KMeans']
metrics = ['psnr', 'ssim', 'dE00', 'lpips']
mlabels = ['PSNR', 'SSIM', 'CIEDE2000', 'LPIPS']

# 每指标在 3 方法均值上归一化到 [0,1]，统一"越高越好"
means = {m: [k64[k64.method == mm][m].mean() for mm in methods] for m in metrics}
norm = {}
for m in metrics:
    v = np.array(means[m])
    if m in ('dE00', 'lpips'):  # 越低越好 -> 反转
        v = v.max() - v
    rng = v.max() - v.min()
    norm[m] = v / rng if rng > 0 else v

fig, ax = plt.subplots(figsize=(6.5, 3.4))
x = np.arange(len(methods))
w = 0.2
colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
for i, (m, ml) in enumerate(zip(metrics, mlabels)):
    ax.bar(x + (i - 1.5) * w, norm[m], w, label=ml, color=colors[i])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Normalized score (higher = better)')
ax.set_title('Multi-metric comparison, K = 64 (Kodak)')
ax.legend(ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12))
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig13_multimetric_bars.pdf'))
plt.savefig(os.path.join(FIG, 'fig13_multimetric_bars.png'), dpi=150)
plt.close()
print('fig13 已重绘')
for m, ml in zip(metrics, mlabels):
    print(f'  {ml}: ' + ', '.join(f'{lab}={means[m][j]:.4f}' for j, lab in enumerate(labels)))
