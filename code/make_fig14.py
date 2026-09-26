# -*- coding: utf-8 -*-
"""重绘 fig14：CE-KMeans 收敛曲线（左：mean pixel-center CIEDE2000；右：max center displacement）。
数据源：convergence_trace.npy（convergence_analysis.py 产出，完整 CIEDE2000 实现）。"""

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
                     'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})
D = f'{_ROOT}/data'
FIG = f'{_ROOT}/figures'

tr = np.load(os.path.join(D, 'convergence_trace.npy'), allow_pickle=True)
print('npy shape:', tr.shape)
d1, d6, d23, move = tr[0], tr[1], tr[2], tr[3]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
it = np.arange(1, len(d1) + 1)

axes[0].plot(it, d1, '-o', ms=3, label='kodim01')
axes[0].plot(it[:len(d6)], d6, '-s', ms=3, label='kodim06')
axes[0].plot(it[:len(d23)], d23, '-^', ms=3, label='kodim23')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Mean pixel-center $\\Delta E_{00}$')
axes[0].set_title('(a) Convergence of objective')
axes[0].legend(frameon=False, fontsize=8)

axes[1].plot(np.arange(1, len(move) + 1), move, '-o', ms=3, color='#c44e52')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Max center displacement (Lab)')
axes[1].set_title('(b) Center movement')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig14_convergence.pdf'))
plt.savefig(os.path.join(FIG, 'fig14_convergence.png'), dpi=150)
plt.close()
print('fig14 已重绘')
print(f'  kodim01 dE00: {d1[0]:.3f} -> {d1[-1]:.3f}')
print(f'  kodim06 dE00: {d6[0]:.3f} -> {d6[-1]:.3f}')
print(f'  kodim23 dE00: {d23[0]:.3f} -> {d23[-1]:.3f}')
