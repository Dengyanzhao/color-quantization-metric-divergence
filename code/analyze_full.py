# -*- coding: utf-8 -*-
"""Full analysis: merge baseline + PW results, generate paper tables & figures."""

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

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})
CB = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB', '#000000']

BASE = pd.read_csv(f'{_ROOT}/data/results.csv')
PW = pd.read_csv(f'{_ROOT}/data/pw_results.csv')
ALL = pd.concat([BASE, PW], ignore_index=True)
N = [16, 32, 64, 128, 256]
OUT_FIG = f'{_ROOT}/figures/'
OUT_DATA = f'{_ROOT}/data/'

LABEL = {
    'mediancut': 'Median Cut', 'maxcoverage': 'Max Coverage', 'fast': 'Fast Octree',
    'kmeans': 'K-means (RGB)', 'som': 'SOM',
    'pw_lab': 'PW-Lab', 'pw_lab_now': 'Lab (no weight)', 'pw_rgb_w': 'PW-RGB',
}

# ============ Table 2: proposed vs baselines at K=64 ============
print('=== Table 2 (K=64): all methods ===')
t64 = ALL[ALL.n_colors == 64].groupby('method')[['psnr', 'ssim', 'time_ms', 'png_bytes']].mean()
t64.columns = ['PSNR', 'SSIM', 'Time(ms)', 'Bytes']
print(t64.round(3).to_string())

# ============ Pairwise: PW vs K-means ============
print('\n=== PW variants vs K-means baseline (mean delta) ===')
for k in N:
    km = BASE[(BASE.method == 'kmeans') & (BASE.n_colors == k)].set_index('image')
    for v in ['pw_lab', 'pw_rgb_w']:
        pv = PW[(PW.method == v) & (PW.n_colors == k)].set_index('image')
        d_p = (pv.psnr - km.psnr).mean()
        d_s = (pv.ssim - km.ssim).mean()
        print(f'  K={k:4d} {v:<10}: dPSNR={d_p:+.2f} dB  dSSIM={d_s:+.4f}')

# wins count
print('\n=== Win counts vs K-means (K=64) ===')
km64 = BASE[(BASE.method == 'kmeans') & (BASE.n_colors == 64)].set_index('image')
for v in ['pw_lab', 'pw_rgb_w']:
    pv = PW[(PW.method == v) & (PW.n_colors == 64)].set_index('image')
    w_p = (pv.psnr > km64.psnr).sum()
    w_s = (pv.ssim > km64.ssim).sum()
    print(f'  {v:<10}: PSNR wins {w_p}/24, SSIM wins {w_s}/24')

# ============ Figures: proposed vs baselines ============
# Fig 5: SSIM vs K (all methods incl. PW)
plt.figure(figsize=(5.5, 3.8))
for i, m in enumerate(LABEL):
    sub = ALL[ALL.method == m].groupby('n_colors')['ssim'].mean()
    ls = '--' if m.startswith('pw_') else '-'
    plt.plot(N, sub, ls, color=CB[i], label=LABEL[m], linewidth=1.5, markersize=4)
plt.xlabel('Number of Colors K')
plt.ylabel('SSIM')
plt.xscale('log', base=2)
plt.xticks(N, N)
plt.legend(frameon=False, ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(OUT_FIG + 'fig5_ssim_all.pdf')
plt.close()

# Fig 6: rate-distortion (SSIM vs bytes) K=64 highlight
plt.figure(figsize=(5.5, 3.8))
for i, m in enumerate(LABEL):
    sub = ALL[ALL.method == m].groupby('n_colors')[['ssim', 'png_bytes']].mean()
    ls = '--' if m.startswith('pw_') else '-'
    plt.plot(sub['png_bytes'] / 1024, sub['ssim'], ls, color=CB[i],
             label=LABEL[m], linewidth=1.5, markersize=4)
plt.xlabel('Palette-PNG Size (KB)')
plt.ylabel('SSIM')
plt.legend(frameon=False, ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(OUT_FIG + 'fig6_rd_all.pdf')
plt.close()

# save merged summary
ALL.groupby(['method', 'n_colors']).agg(
    psnr=('psnr', 'mean'), ssim=('ssim', 'mean'),
    time_ms=('time_ms', 'mean'), png_bytes=('png_bytes', 'mean')).round(3).to_csv(
    OUT_DATA + 'all_methods_summary.csv')
print('\nSaved merged summary + figures.')
