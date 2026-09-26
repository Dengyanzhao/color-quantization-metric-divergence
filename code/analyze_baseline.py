# -*- coding: utf-8 -*-
"""Analyze baseline experiment results: summary tables + figures."""

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
    'font.family': 'sans-serif',
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})
CB = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']

df = pd.read_csv(f'{_ROOT}/data/results.csv')
OUT_FIG = f'{_ROOT}/figures/'
OUT_DATA = f'{_ROOT}/data/'

METHOD_LABEL = {
    'mediancut': 'Median Cut',
    'maxcoverage': 'Max Coverage',
    'fast': 'Fast Octree',
    'kmeans': 'K-means (RGB)',
    'som': 'SOM',
}

# ---- summary table: mean ± std by (method, n_colors) ----
summary = df.groupby(['method', 'n_colors']).agg(
    psnr_mean=('psnr', 'mean'), psnr_std=('psnr', 'std'),
    ssim_mean=('ssim', 'mean'), ssim_std=('ssim', 'std'),
    time_mean=('time_ms', 'mean'), time_std=('time_ms', 'std'),
    bytes_mean=('png_bytes', 'mean'), bytes_std=('png_bytes', 'std'),
).round(3).reset_index()
summary.to_csv(OUT_DATA + 'baseline_summary.csv', index=False)
print(summary.to_string(index=False))

# ---- overall mean table for paper (Table 1) ----
print('\n=== Paper Table 1: mean over all images ===')
pivot_psnr = df.pivot_table(index='method', columns='n_colors', values='psnr', aggfunc='mean').round(2)
pivot_ssim = df.pivot_table(index='method', columns='n_colors', values='ssim', aggfunc='mean').round(4)
pivot_time = df.pivot_table(index='method', columns='n_colors', values='time_ms', aggfunc='mean').round(1)
pivot_bytes = df.pivot_table(index='method', columns='n_colors', values='png_bytes', aggfunc='mean').round(0)
print('PSNR:'); print(pivot_psnr.to_string())
print('\nSSIM:'); print(pivot_ssim.to_string())
print('\nTime (ms):'); print(pivot_time.to_string())
print('\nBytes:'); print(pivot_bytes.to_string())

# ---- figures ----
N = [16, 32, 64, 128, 256]
for m in METHOD_LABEL:
    sub = df[df.method == m].groupby('n_colors')[['psnr', 'ssim', 'time_ms', 'png_bytes']].mean()

    # Fig: PSNR vs colors
    plt.figure(figsize=(5.5, 3.8))
    for i, m2 in enumerate(METHOD_LABEL):
        sub2 = df[df.method == m2].groupby('n_colors')['psnr'].mean()
        plt.plot(N, sub2, 'o-', color=CB[i], label=METHOD_LABEL[m2], linewidth=1.5, markersize=4)
    plt.xlabel('Number of Colors K')
    plt.ylabel('PSNR (dB)')
    plt.xscale('log', base=2)
    plt.xticks(N, N)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_FIG + 'fig1_psnr_vs_colors.pdf')
    plt.close()

    # Fig: SSIM vs colors
    plt.figure(figsize=(5.5, 3.8))
    for i, m2 in enumerate(METHOD_LABEL):
        sub2 = df[df.method == m2].groupby('n_colors')['ssim'].mean()
        plt.plot(N, sub2, 'o-', color=CB[i], label=METHOD_LABEL[m2], linewidth=1.5, markersize=4)
    plt.xlabel('Number of Colors K')
    plt.ylabel('SSIM')
    plt.xscale('log', base=2)
    plt.xticks(N, N)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_FIG + 'fig2_ssim_vs_colors.pdf')
    plt.close()

    # Fig: time comparison
    plt.figure(figsize=(5.5, 3.8))
    for i, m2 in enumerate(METHOD_LABEL):
        sub2 = df[df.method == m2].groupby('n_colors')['time_ms'].mean()
        plt.plot(N, sub2, 'o-', color=CB[i], label=METHOD_LABEL[m2], linewidth=1.5, markersize=4)
    plt.xlabel('Number of Colors K')
    plt.ylabel('Time (ms, log scale)')
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xticks(N, N)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_FIG + 'fig3_time_vs_colors.pdf')
    plt.close()

    # Fig: rate-distortion style (SSIM vs file size)
    plt.figure(figsize=(5.5, 3.8))
    for i, m2 in enumerate(METHOD_LABEL):
        sub2 = df[df.method == m2].groupby('n_colors')[['ssim', 'png_bytes']].mean()
        plt.plot(sub2['png_bytes'] / 1024, sub2['ssim'], 'o-', color=CB[i],
                 label=METHOD_LABEL[m2], linewidth=1.5, markersize=4)
    plt.xlabel('File Size (KB)')
    plt.ylabel('SSIM')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_FIG + 'fig4_ssim_vs_size.pdf')
    plt.close()

print('\nFigures saved to', OUT_FIG)
