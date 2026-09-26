# -*- coding: utf-8 -*-
"""方案 A：指标一致性量化（Kodak, K-means/PW-Lab/CE-KMeans × 5 色数）。
可复现分析：读 metric_full_matrix.csv（psnr/ssim/dE00/lpips 合并宽表），
输出 (1) 4x4 Kendall tau 热图 fig15；(2) 决策翻转率统计。
依赖：eval_lpips.py 与 ce_kmeans.py 产出的宽表。"""

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
from scipy.stats import kendalltau

D = f'{_ROOT}/data'
FIG = f'{_ROOT}/figures'
METRICS = ['psnr', 'ssim', 'dE00', 'lpips']
LABELS = ['PSNR', 'SSIM', 'CIEDE2000', 'LPIPS']
# dE00/lpips 越低越好 -> 取负统一为"越高越好"方向
NEGATE = {'dE00', 'lpips'}


def load():
    df = pd.read_csv(os.path.join(D, 'metric_full_matrix.csv'))
    return df.dropna(subset=METRICS)


def kendall_matrix(df):
    """逐图在 15 个(方法×色数)组合上算 4 指标两两 Kendall tau，再对图平均。"""
    n = len(METRICS)
    acc = np.zeros((n, n))
    cnt = 0
    for img, sub in df.groupby('image'):
        if len(sub) < 5:
            continue
        cnt += 1
        vals = {}
        for m in METRICS:
            v = sub[m].values
            vals[m] = -v if m in NEGATE else v
        for i in range(n):
            for j in range(n):
                acc[i, j] += kendalltau(vals[METRICS[i]], vals[METRICS[j]])[0]
    return acc / cnt, cnt


def flip_rate(df, k=None):
    """逐图：按 CIEDE2000 选最优方法 vs 按 LPIPS 选最优方法，不一致的比例。"""
    sub = df if k is None else df[df.n_colors == k]
    flip = 0
    total = 0
    for img, g in sub.groupby('image'):
        if len(g) < 2:
            continue
        total += 1
        best_de = g.loc[g.dE00.idxmin(), 'method']
        best_lp = g.loc[g.lpips.idxmin(), 'method']
        if best_de != best_lp:
            flip += 1
    return flip, total


def main():
    df = load()
    print(f'宽表行数: {len(df)}, 图数: {df.image.nunique()}')

    tau, cnt = kendall_matrix(df)
    print(f'\n=== Kendall tau（{cnt} 图平均，正=同向排序）===')
    print('           ' + ' '.join(f'{l:>10}' for l in LABELS))
    for i, li in enumerate(LABELS):
        print(f'{li:>10} ' + ' '.join(f'{tau[i, j]:>10.3f}' for j in range(len(LABELS))))
    np.savetxt(os.path.join(D, 'metric_kendall_tau.csv'), tau,
               delimiter=',', fmt='%.4f', header=','.join(LABELS), comments='')

    # 热图
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(tau, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha='right')
    ax.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f'{tau[i, j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title("Kendall $\\tau$ between metrics\n(averaged over images)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig15_metric_tau_heatmap.pdf'))
    plt.savefig(os.path.join(FIG, 'fig15_metric_tau_heatmap.png'), dpi=150)
    plt.close()
    print('\n热图已保存: fig15_metric_tau_heatmap.pdf')

    # 翻转率
    f_all, t_all = flip_rate(df)
    f64, t64 = flip_rate(df, k=64)
    print(f'\n=== 决策翻转率（CIEDE2000 最优 vs LPIPS 最优）===')
    print(f'全色数网格: {f_all}/{t_all} = {f_all/t_all*100:.0f}%')
    print(f'K=64 口径:  {f64}/{t64} = {f64/t64*100:.0f}%')


if __name__ == '__main__':
    main()
