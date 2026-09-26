# -*- coding: utf-8 -*-
"""Adaptive palette size selection experiment.
Features (color entropy, edge density, gradient energy) -> optimal K.
Efficiency curve: K* = argmax marginal SSIM gain per byte.
LOO cross-validation with Random Forest."""

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

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import sobel
from sklearn.ensemble import RandomForestRegressor
import os, sys

KODAK_DIR = f'{_DATASETS}/kodak'
BASE = pd.read_csv(f'{_ROOT}/data/results.csv')
N_GRID = [16, 32, 64, 128, 256]


def extract_features(img):
    """Color entropy (16^3 hist), edge density, gradient energy."""
    arr = np.array(img.convert('RGB'))
    h, w, _ = arr.shape

    # color entropy on 16^3 quantized histogram
    q = (arr // 16).astype(np.int32)
    idx = (q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]).ravel()
    hist = np.bincount(idx, minlength=4096).astype(np.float64)
    hist = hist[hist > 0]
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist))

    # edge density & gradient energy
    gray = np.array(img.convert('L')).astype(np.float64)
    g = sobel(gray)
    tau = np.percentile(g, 90)
    edge_density = (g > tau).mean()
    grad_energy = np.mean(g ** 2)
    return entropy, edge_density, grad_energy


def efficiency_best_k(row_df):
    """K* via marginal SSIM-per-byte efficiency."""
    best_k, best_e = None, -np.inf
    prev = row_df[row_df.n_colors == N_GRID[0]].iloc[0]
    prev_s, prev_b = prev.ssim, prev.png_bytes
    for k in N_GRID[1:]:
        cur = row_df[row_df.n_colors == k].iloc[0]
        dq = cur.ssim - prev_s
        db = max(cur.png_bytes - prev_b, 1)
        e = dq / db
        if e > best_e:
            best_e, best_k = e, k
        prev_s, prev_b = cur.ssim, cur.png_bytes
    return best_k if best_k is not None else N_GRID[0]


def main():
    feats, kstars, names = [], [], []
    for fn in sorted(os.listdir(KODAK_DIR)):
        if not fn.endswith('.png'):
            continue
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        f = extract_features(img)
        row_df = BASE[BASE.image == fn]
        # use K-means (RGB) efficiency curve as the reference
        k = efficiency_best_k(row_df[row_df.method == 'kmeans'])
        feats.append(f)
        kstars.append(k)
        names.append(fn)
        sys.stdout.write(f'{fn}: H={f[0]:.2f} e={f[1]:.3f} G={f[2]:.0f} -> K*={k}\n')

    X = np.array(feats)
    y = np.array(kstars)
    df = pd.DataFrame({'image': names, 'H': X[:, 0], 'edge': X[:, 1],
                       'grad': X[:, 2], 'Kstar': y})
    df.to_csv(f'{_ROOT}/data/adaptive_features.csv', index=False)

    # LOO cross-validation
    preds = []
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[mask], y[mask])
        p = rf.predict(X[i:i + 1])[0]
        preds.append(min(N_GRID, key=lambda k: abs(k - p)))
    acc = np.mean(np.array(preds) == y)
    mae = np.mean(np.abs(np.array(preds) - y))
    print(f'\nLOO accuracy (exact match): {acc:.3f}')
    print(f'LOO MAE (colors): {mae:.1f}')
    print('K* distribution:', {k: int((y == k).sum()) for k in N_GRID})

    # Benefit analysis: adaptive vs fixed baselines (using baseline kmeans data)
    print('\n=== Benefit: adaptive K* vs fixed K (SSIM / bytes, mean over images) ===')
    rows = []
    for fn in names:
        row_df = BASE[(BASE.image == fn) & (BASE.method == 'kmeans')]
        s_opt = row_df[row_df.n_colors == y[names.index(fn)]].iloc[0]
        row = {'image': fn, 'ssim_adaptive': s_opt.ssim, 'bytes_adaptive': s_opt.png_bytes}
        for k in N_GRID:
            r = row_df[row_df.n_colors == k].iloc[0]
            row[f'ssim_k{k}'] = r.ssim
            row[f'bytes_k{k}'] = r.png_bytes
        rows.append(row)
    bdf = pd.DataFrame(rows)
    bdf.to_csv(f'{_ROOT}/data/adaptive_benefit.csv', index=False)
    print(bdf[['ssim_adaptive'] + [f'ssim_k{k}' for k in N_GRID]].mean().round(4).to_string())
    print()
    print(bdf[['bytes_adaptive'] + [f'bytes_k{k}' for k in N_GRID]].mean().round(0).to_string())


if __name__ == '__main__':
    main()