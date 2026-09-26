# -*- coding: utf-8 -*-
"""Large-scale validation on BSDS300 test set (100 images).
Part 1: CE-KMeans vs K-means vs PW-Lab at K=64 (dE00/PSNR/SSIM).
Part 2: training-free iterative palette search (tau=2.5)."""

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
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim_fn
from sklearn.cluster import KMeans
import os, sys, io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ce_kmeans import ce_kmeans, iterative_palette, psnr
from pw_kmeans_v3 import pw_kmeans

BSD_DIR = f'{_DATASETS}/bsds300'
K = 64
TAU = 2.5


def quantize_kmeans_sk(img, k):
    arr = np.array(img).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(42)
    km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
    km.fit(arr[rng.choice(len(arr), min(50000, len(arr)), replace=False)])
    out = km.cluster_centers_.round().astype(np.uint8)[km.predict(arr)]
    return Image.fromarray(out.reshape(img.size[1], img.size[0], 3))


def main():
    files = sorted([f for f in os.listdir(BSD_DIR) if f.endswith(('.jpg', '.png'))])
    rows1, rows2 = [], []
    for fn in files:
        img = Image.open(os.path.join(BSD_DIR, fn)).convert('RGB')
        arr = np.array(img)
        lab_ref = rgb2lab(arr.astype(np.float64) / 255.0)

        # Part 1: K=64 comparison
        outs = {
            'kmeans_rgb': quantize_kmeans_sk(img, K),
            'pw_lab': pw_kmeans(img, K, use_lab=True, weight_power=0.5),
            'ce_kmeans': ce_kmeans(img, K),
        }
        for m, out in outs.items():
            oa = np.array(out)
            rows1.append({'image': fn, 'method': m,
                          'psnr': round(psnr(arr, oa), 3),
                          'ssim': round(float(ssim_fn(arr, oa, channel_axis=2, data_range=255)), 4),
                          'dE00': round(float(deltaE_ciede2000(lab_ref,
                                  rgb2lab(oa.astype(np.float64) / 255.0)).mean()), 3)})

        # Part 2: iterative search
        k, out, d = iterative_palette(img, TAU)
        oa = np.array(out)
        bio = io.BytesIO()
        out.convert('P', palette=Image.Palette.ADAPTIVE, colors=k).save(bio, format='PNG')
        rows2.append({'image': fn, 'K_iter': k, 'dE00_iter': round(d, 3),
                      'bytes_iter': bio.tell(), 'psnr_iter': round(psnr(arr, oa), 2)})
        sys.stdout.write(f'{fn}: dE00 kmeans={rows1[-3]["dE00"]} pw={rows1[-2]["dE00"]} '
                         f'ce={rows1[-1]["dE00"]} | iter K={k} bytes={bio.tell()}\n')
        sys.stdout.flush()

    df1 = pd.DataFrame(rows1)
    df2 = pd.DataFrame(rows2)
    df1.to_csv(f'{_ROOT}/data/bsd100_methods.csv', index=False)
    df2.to_csv(f'{_ROOT}/data/bsd100_iterative.csv', index=False)

    print('\n=== BSDS100 K=64 comparison (mean over 100 images) ===')
    for m in ['kmeans_rgb', 'pw_lab', 'ce_kmeans']:
        sub = df1[df1.method == m]
        print(f'{m:<12} dE00={sub.dE00.mean():.3f} PSNR={sub.psnr.mean():.2f} SSIM={sub.ssim.mean():.4f}')

    from scipy.stats import wilcoxon
    km_d = df1[df1.method == 'kmeans_rgb'].sort_values('image').dE00.values
    pw_d = df1[df1.method == 'pw_lab'].sort_values('image').dE00.values
    ce_d = df1[df1.method == 'ce_kmeans'].sort_values('image').dE00.values
    w1, p1 = wilcoxon(ce_d, km_d)
    w2, p2 = wilcoxon(ce_d, pw_d)
    print(f'\nWilcoxon: CE vs K-means p={p1:.2e} (wins {(ce_d < km_d).sum()}/100)')
    print(f'          CE vs PW-Lab  p={p2:.2e} (wins {(ce_d < pw_d).sum()}/100)')

    print('\n=== BSDS100 iterative search (tau=%.1f) ===' % TAU)
    print(f'K distribution: {df2.K_iter.value_counts().sort_index().to_dict()}')
    print(f'meet rate: {(df2.dE00_iter <= TAU).mean():.3f}, avg bytes: {df2.bytes_iter.mean():.0f}')
    # fixed-K reference on BSDS: approximate from part-1 CE data at K=64
    sub64 = df1[df1.method == 'ce_kmeans']
    print(f'fixed K=64 reference: dE00={sub64.dE00.mean():.3f} (meet rate {(sub64.dE00 <= TAU).mean():.3f})')


if __name__ == '__main__':
    main()