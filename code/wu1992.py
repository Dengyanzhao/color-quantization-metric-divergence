# -*- coding: utf-8 -*-
"""Wu 1992 PCA splitting quantization (pixel-space version).
Direct pixel-space PCA splitting, no histogram approximation.
"""

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
from PIL import Image
import sys, os, time
from skimage.color import rgb2lab, deltaE_ciede2000


def pca_split(pixels, indices):
    """Split pixels at indices along principal component.
    Returns (left_indices, right_indices) or None."""
    n = len(indices)
    if n < 2:
        return None
    sub = pixels[indices]
    mean = sub.mean(axis=0)
    centered = sub - mean
    cov = centered.T @ centered / (n - 1)
    w, v = np.linalg.eigh(cov)
    pc = v[:, -1]  # PC1
    proj = centered @ pc
    # try mean split first, then search around mean for optimal
    # use sorted projection
    order = np.argsort(proj)
    sorted_proj = proj[order]
    sorted_indices = indices[order]
    # accumulate mean squared error
    cum_sum = np.cumsum(sub[order], axis=0)
    cum_sum2 = np.cumsum(sub[order] ** 2, axis=0)
    total_n = n
    total_sum = cum_sum[-1]
    total_sum2 = cum_sum2[-1]
    best_err = float('inf')
    best_k = -1
    for k in range(1, n):
        n1 = k
        n2 = total_n - k
        if n1 < 1 or n2 < 1:
            continue
        s1 = cum_sum[k - 1]
        s2 = total_sum - s1
        err1 = cum_sum2[k - 1].sum() - (s1 * s1 / n1).sum()
        err2 = (cum_sum2[-1] - cum_sum2[k - 1]).sum() - (s2 * s2 / n2).sum()
        err = err1 + err2
        if err < best_err:
            best_err = err
            best_k = k
    if best_k <= 0 or best_k >= n:
        return None
    return sorted_indices[:best_k], sorted_indices[best_k:]


def wu_quantize_pixel(img, K):
    """Wu-1992-style PCA splitting quantization (pixel space)."""
    arr = np.array(img).astype(np.float64)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)
    n = len(pixels)
    # priority queue: (error, indices) — use list of tuples, sort by error
    # each leaf = indices array
    leaves = [(np.var(pixels, axis=0).sum(), np.arange(n))]
    while len(leaves) < K:
        # find leaf with largest variance
        best_idx = -1
        best_var = -1
        for i, (var, idx) in enumerate(leaves):
            if var > best_var:
                best_var = var
                best_idx = i
        if best_idx < 0:
            break
        var, idx = leaves.pop(best_idx)
        split = pca_split(pixels, idx)
        if split is None:
            leaves.append((var, idx))
            break
        l_idx, r_idx = split
        l_var = np.var(pixels[l_idx], axis=0).sum()
        r_var = np.var(pixels[r_idx], axis=0).sum()
        leaves.append((l_var, l_idx))
        leaves.append((r_var, r_idx))
    # compute palette (mean per leaf)
    palette = []
    for _, idx in leaves:
        if len(idx) > 0:
            palette.append(pixels[idx].mean(axis=0))
    palette = np.array(palette).round().astype(np.uint8)
    # assign all pixels
    flat = arr.reshape(-1, 3).astype(np.float32)
    labels = np.argmin(np.linalg.norm(flat[:, None, :] - palette[None, :, :].astype(np.float32), axis=2), axis=1)
    return Image.fromarray(palette[labels].reshape(h, w, 3))


def main():
    KODAK = f'{_DATASETS}/kodak'
    K = 64
    files = sorted([f for f in os.listdir(KODAK) if f.endswith('.png')])
    rows = []
    for fn in files:
        img = Image.open(f'{KODAK}/{fn}').convert('RGB')
        arr = np.array(img)
        lab_ref = rgb2lab(arr.astype(np.float64) / 255.0)
        t0 = time.time()
        out = wu_quantize_pixel(img, K)
        dt = time.time() - t0
        oa = np.array(out)
        mse = np.mean((arr.astype(float) - oa.astype(float)) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse)
        d = float(deltaE_ciede2000(lab_ref, rgb2lab(oa.astype(np.float64) / 255.0)).mean())
        rows.append((fn, psnr, d, dt))
        sys.stdout.write('%s: Wu PSNR=%.2f dE00=%.3f t=%.1fs\n' % (fn, psnr, d, dt))
        sys.stdout.flush()
    print('\n=== Wu 1992 (pixel-space PCA) Kodak 24, K=64 ===')
    print('PSNR: %.2f  dE00: %.3f  平均耗时: %.1fs' % (
        np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows])))
    print('参考: CE-KMeans PSNR=33.2 dE00=2.155 | K-means PSNR=35.85 dE00=2.507')


if __name__ == '__main__':
    main()