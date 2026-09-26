# -*- coding: utf-8 -*-
"""Control experiment: K-means with median-cut init + single run (fair comparison to PW variants).
Then: Wilcoxon signed-rank test on CIEDE2000 (PW-Lab vs K-means-rgb)."""

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
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.filters import sobel
import os, sys, io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_kmeans_v3 import psnr, assign_chunked, weighted_kmeans

KODAK_DIR = f'{_DATASETS}/kodak'
K = 64


def kmeans_mc_init(img, k):
    """K-means in RGB with median-cut initialization, single run (fair to PW)."""
    arr = np.array(img).astype(np.float32)
    mc = img.quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    centers = np.array(mc.getpalette()[:k*3]).reshape(-1, 3).astype(np.float32)
    pixels = arr.reshape(-1, 3)
    idx = np.random.default_rng(42).choice(len(pixels), 50000, replace=False)
    tr = pixels[idx]
    labels = assign_chunked(tr, centers, chunk=10000)
    for _ in range(20):
        new = np.zeros_like(centers)
        for kk in range(k):
            m = (labels == kk)
            if m.sum() > 0:
                new[kk] = tr[m].mean(axis=0)
            else:
                new[kk] = centers[kk]
        if np.allclose(centers, new, atol=1e-4):
            break
        centers = new
        labels = assign_chunked(tr, centers, chunk=10000)
    out = centers.round().astype(np.uint8)[assign_chunked(pixels, centers)]
    return Image.fromarray(out.reshape(img.size[1], img.size[0], 3))


def main():
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    psnr_mc, ssim_mc, d00_mc = [], [], []
    psnr_km, ssim_km, d00_km = [], [], []
    for fn in files:
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        arr = np.array(img)
        out_mc = kmeans_mc_init(img, K)
        km = KMeans(n_clusters=K, n_init=3, random_state=42, max_iter=100)
        km.fit(arr.reshape(-1, 3)[np.random.default_rng(42).choice(arr.shape[0]*arr.shape[1], 50000, replace=False)])
        out_km = Image.fromarray(km.cluster_centers_.round().astype(np.uint8)[km.predict(arr.reshape(-1, 3))].reshape(arr.shape))
        from skimage.metrics import structural_similarity as ssim_fn
        for tag, out, pl, sl, dl in [('mc', out_mc, psnr_mc, ssim_mc, d00_mc),
                                     ('km', out_km, psnr_km, ssim_km, d00_km)]:
            oa = np.array(out)
            pl.append(psnr(arr, oa))
            sl.append(float(ssim_fn(arr, oa, channel_axis=2, data_range=255)))
            dl.append(float(deltaE_ciede2000(rgb2lab(arr/255.0), rgb2lab(oa/255.0)).mean()))
        print(f'{fn}: MC-init PSNR={psnr_mc[-1]:.2f} dE00={d00_mc[-1]:.3f} | KMeans++ PSNR={psnr_km[-1]:.2f} dE00={d00_km[-1]:.3f}')

    print('\n=== Fairness check (K=64) ===')
    print(f'K-means(MC-init, 1 run):  PSNR={np.mean(psnr_mc):.2f}  dE00={np.mean(d00_mc):.3f}')
    print(f'K-means(kmeans++, 3 runs): PSNR={np.mean(psnr_km):.2f}  dE00={np.mean(d00_km):.3f}')

    # Wilcoxon signed-rank: PW-Lab dE00 (from eval) vs K-means dE00
    from scipy.stats import wilcoxon
    pwlab_d00 = [1.880,1.030,2.103,2.172,3.199,1.922,2.028,2.905,1.808,2.002,2.151,1.702,
                 2.654,2.669,2.227,1.718,1.866,2.923,2.054,1.586,2.196,2.896,3.024,2.655]
    w, p = wilcoxon(pwlab_d00, d00_km)
    print(f'\nWilcoxon signed-rank (PW-Lab vs K-means dE00): W={w}, p={p:.6f}')
    # also PSNR wilcoxon
    wp, pp = wilcoxon(pwlab_d00, d00_mc)
    print(f'Wilcoxon (PW-Lab vs K-means MC-init dE00): W={wp}, p={pp:.6f}')
    np.save(f'{_ROOT}/data/fairness.npy', np.array([psnr_mc, psnr_km, d00_mc, d00_km]))

if __name__ == '__main__':
    main()
