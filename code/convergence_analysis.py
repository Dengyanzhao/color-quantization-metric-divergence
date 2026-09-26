# -*- coding: utf-8 -*-
"""CE-KMeans 收敛性分析：记录每轮迭代的目标函数值（平均像素-色心 dE00）。
输出：收敛曲线数据 + Kodak 24 平均迭代轮数统计。"""

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

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
sys.path.insert(0, f'{_CODE}')
from ce_kmeans import ce_kmeans, lab2lch, assign_ce_full_njit

KODAK = f'{_DATASETS}/kodak'


def convergence_trace(img, K=64, max_iter=20):
    """运行 CE-KMeans 并记录每轮的平均像素-色心 dE00 与中心位移。"""
    arr = np.array(img).astype(np.float64) / 255.0
    lab = rgb2lab(arr)
    h, w, _ = lab.shape
    pixels = lab.reshape(-1, 3).astype(np.float32)
    mc = img.quantize(colors=K, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    centers_rgb = np.array(mc.getpalette()[:K * 3]).reshape(-1, 3).astype(np.float64)
    centers = rgb2lab(centers_rgb[None, :, :] / 255.0)[0].astype(np.float32)
    RNG = np.random.default_rng(42)
    idx = RNG.choice(len(pixels), min(50000, len(pixels)), replace=False)
    tr = pixels[idx]
    tr_lch = lab2lch(tr).astype(np.float32)
    p_lch = lab2lch(pixels).astype(np.float32)

    trace = {'dE00': [], 'move': []}
    for it in range(max_iter):
        labels = assign_ce_full_njit(tr.astype(np.float64), centers.astype(np.float64))
        # 平均 dE00 用完整 CIEDE2000（skimage，与评估口径一致）
        d = deltaE_ciede2000(tr.astype(np.float64), centers.astype(np.float64)[labels])
        trace['dE00'].append(float(np.mean(d)))

        # M 步
        new_centers = np.zeros_like(centers)
        for k in range(K):
            m = (labels == k)
            if m.sum() > 0:
                new_centers[k] = tr[m].mean(axis=0)
            else:
                new_centers[k] = centers[k]
        trace['move'].append(float(np.abs(new_centers - centers).max()))
        if np.allclose(centers, new_centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers
    return trace


def main():
    files = sorted([f for f in os.listdir(KODAK) if f.endswith('.png')])
    # 收敛曲线（3 张代表图）
    traces = {}
    for fn in ['kodim01.png', 'kodim06.png', 'kodim23.png']:
        img = Image.open(f'{KODAK}/{fn}').convert('RGB')
        traces[fn] = convergence_trace(img)
        print(f'{fn}: 迭代 {len(traces[fn]["dE00"])} 轮, '
              f'dE00 {traces[fn]["dE00"][0]:.3f} -> {traces[fn]["dE00"][-1]:.3f}')

    # Kodak 24 平均迭代轮数
    iters = []
    for fn in files:
        img = Image.open(f'{KODAK}/{fn}').convert('RGB')
        tr = convergence_trace(img)
        iters.append(len(tr['dE00']))
    print(f'\nKodak 24 平均收敛轮数: {np.mean(iters):.1f} (范围 {min(iters)}-{max(iters)})')

    # 保存数据
    np.save(f'{_ROOT}/data/convergence_trace.npy',
            np.array([traces['kodim01.png']['dE00'], traces['kodim06.png']['dE00'],
                      traces['kodim23.png']['dE00'], traces['kodim01.png']['move']]))
    print('收敛数据已保存')


if __name__ == '__main__':
    main()