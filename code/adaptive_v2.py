# -*- coding: utf-8 -*-
"""Adaptive palette size selection v2: quality-constrained minimal K.
For each image: K* = min K such that SSIM(K) >= tau (target quality).
This yields content-dependent K* across the grid, then RF learns features->K*."""

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
TAU = 0.95  # target SSIM


def extract_features(img):
    arr = np.array(img.convert('RGB'))
    h, w, _ = arr.shape
    q = (arr // 16).astype(np.int32)
    idx = (q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]).ravel()
    hist = np.bincount(idx, minlength=4096).astype(np.float64)
    hist = hist[hist > 0]
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist))
    gray = np.array(img.convert('L')).astype(np.float64)
    g = sobel(gray)
    tau_edge = np.percentile(g, 90)
    edge_density = (g > tau_edge).mean()
    grad_energy = np.mean(g ** 2)
    return entropy, edge_density, grad_energy


def constrained_k(row_df, tau, method='kmeans'):
    """Minimal K meeting SSIM>=tau; if none meets, largest K."""
    row_df = row_df[row_df.method == method].sort_values('n_colors')
    for _, r in row_df.iterrows():
        if r.ssim >= tau:
            return int(r.n_colors)
    return int(row_df.iloc[-1].n_colors)


def main():
    feats, kstars, names = [], [], []
    for fn in sorted(os.listdir(KODAK_DIR)):
        if not fn.endswith('.png'):
            continue
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        f = extract_features(img)
        k = constrained_k(BASE[BASE.image == fn], TAU)
        feats.append(f)
        kstars.append(k)
        names.append(fn)
        sys.stdout.write(f'{fn}: H={f[0]:.2f} e={f[1]:.3f} G={f[2]:.0f} -> K*={k}\n')

    X = np.array(feats)
    y = np.array(kstars)
    pd.DataFrame({'image': names, 'H': X[:, 0], 'edge': X[:, 1],
                  'grad': X[:, 2], 'Kstar': y}).to_csv(
        f'{_ROOT}/data/adaptive_features.csv', index=False)

    print(f'\nK* distribution (tau={TAU}):', {k: int((y == k).sum()) for k in N_GRID})

    # LOO CV
    preds = []
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X[mask], y[mask])
        p = rf.predict(X[i:i + 1])[0]
        preds.append(min(N_GRID, key=lambda k: abs(k - p)))
    preds = np.array(preds)
    print(f'LOO accuracy: {np.mean(preds == y):.3f}, MAE: {np.mean(np.abs(preds - y)):.1f}')

    # Benefit: adaptive (oracle K*) vs fixed K under the SSIM>=tau constraint
    # Metric: bytes used while meeting constraint; lower is better.
    print('\n=== Bytes to meet SSIM>=%.2f: adaptive vs fixed ===' % TAU)
    bdf = pd.DataFrame({'image': names})
    for k in N_GRID:
        bdf[f'bytes_k{k}'] = [int(BASE[(BASE.image == n) & (BASE.method == 'kmeans')
                                & (BASE.n_colors == k)].iloc[0].png_bytes) for n in names]
    bdf['bytes_adaptive_oracle'] = [int(BASE[(BASE.image == n) & (BASE.method == 'kmeans')
                                    & (BASE.n_colors == y[names.index(n)])].iloc[0].png_bytes)
                                    for n in names]
    bdf['bytes_adaptive_pred'] = [int(BASE[(BASE.image == n) & (BASE.method == 'kmeans')
                                   & (BASE.n_colors == preds[names.index(n)])].iloc[0].png_bytes)
                                   for n in names]
    bdf.to_csv(f'{_ROOT}/data/adaptive_benefit.csv', index=False)
    cols = ['bytes_adaptive_oracle', 'bytes_adaptive_pred'] + [f'bytes_k{k}' for k in N_GRID]
    print(bdf[cols].mean().round(0).to_string())
    print('\nAdaptive oracle vs best fixed (min over grid):',
          round(bdf.bytes_adaptive_oracle.mean() / bdf[[f'bytes_k{k}' for k in N_GRID]].min(axis=1).mean(), 3))


if __name__ == '__main__':
    main()