# -*- coding: utf-8 -*-
"""CIEDE2000 perceptual color difference evaluation.
Compares K-means (RGB), PW-Lab, PW-RGB-W at K=64 on Kodak.
Lower mean/percentile DeltaE00 = better perceptual fidelity."""

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
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn.cluster import KMeans
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_kmeans_v3 import pw_kmeans, psnr

KODAK_DIR = f'{_DATASETS}/kodak'
K = 64
methods = ['kmeans_rgb', 'pw_rgb_w', 'pw_lab']


def quantize_kmeans_sk(img, k):
    arr = np.array(img).reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
    km.fit(arr[np.random.default_rng(42).choice(len(arr), 50000, replace=False)])
    out = km.cluster_centers_.round().astype(np.uint8)[km.predict(arr)]
    return Image.fromarray(out.reshape(img.size[1], img.size[0], 3))


def ciede_stats(a_rgb, b_rgb):
    """Mean & 95th percentile CIEDE2000 between two RGB images."""
    la = rgb2lab(a_rgb / 255.0)
    lb = rgb2lab(b_rgb / 255.0)
    d = deltaE_ciede2000(la, lb)
    return float(d.mean()), float(np.percentile(d, 95))


def main():
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    acc = {m: {'d00_mean': [], 'd00_p95': [], 'psnr': []} for m in methods}
    for fn in files:
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        arr = np.array(img)
        outs = {
            'kmeans_rgb': quantize_kmeans_sk(img, K),
            'pw_rgb_w': pw_kmeans(img, K, use_lab=False, weight_power=0.5),
            'pw_lab': pw_kmeans(img, K, use_lab=True, weight_power=0.5),
        }
        line = [fn]
        for m in methods:
            oa = np.array(outs[m])
            dm, dp = ciede_stats(arr, oa)
            p = psnr(arr, oa)
            acc[m]['d00_mean'].append(dm)
            acc[m]['d00_p95'].append(dp)
            acc[m]['psnr'].append(p)
            line.append(f'{m}: dE00={dm:.3f} p95={dp:.3f} psnr={p:.2f}')
        sys.stdout.write(' | '.join(line) + '\n')
        sys.stdout.flush()

    print('\n=== Summary (K=%d) ===' % K)
    print(f"{'method':<12}{'dE00_mean':>12}{'dE00_p95':>12}{'PSNR':>10}")
    for m in methods:
        print(f"{m:<12}{np.mean(acc[m]['d00_mean']):>12.3f}{np.mean(acc[m]['d00_p95']):>12.3f}"
              f"{np.mean(acc[m]['psnr']):>10.2f}")
    # pairwise wins
    import itertools
    for a, b in itertools.combinations(methods, 2):
        w_a = sum(1 for x, y in zip(acc[a]['d00_mean'], acc[b]['d00_mean']) if x < y)
        print(f'dE00: {a} beats {b} on {w_a}/{len(acc[a]["d00_mean"])} images')


if __name__ == '__main__':
    main()