# -*- coding: utf-8 -*-
"""DIV2K validation (100 images) experiments.
HR images resized to 512x512 to control compute cost.
Part 1: CE-KMeans vs K-means vs PW-Lab at K=64 (dE00/PSNR/SSIM).
Part 2: training-free iterative palette search (tau=2.5).
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
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim_fn
from sklearn.cluster import KMeans
import os, sys, io, glob

sys.path.insert(0, f'{_CODE}')
from ce_kmeans import ce_kmeans, iterative_palette, psnr
from pw_kmeans_v3 import pw_kmeans

DIV2K_DIR = f'{_DATASETS}/div2k/DIV2K_valid_HR'
K = 64
TAU = 2.5
RESIZE = (512, 512)


def quantize_kmeans_sk(img, k):
    arr = np.array(img).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(42)
    km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
    km.fit(arr[rng.choice(len(arr), min(50000, len(arr)), replace=False)])
    out = km.cluster_centers_.round().astype(np.uint8)[km.predict(arr)]
    return Image.fromarray(out.reshape(img.size[1], img.size[0], 3))


def main():
    files = sorted(glob.glob(os.path.join(DIV2K_DIR, '*.png')))
    files = [f for f in files if os.path.getsize(f) > 10000]  # 过滤占位符
    print(f'找到 {len(files)} 张 DIV2K 验证图')
    rows1, rows2 = [], []
    for fi, fn in enumerate(files):
        img = Image.open(fn).convert('RGB')
        img = img.resize(RESIZE, Image.LANCZOS)  # 统一 512x512
        arr = np.array(img)
        lab_ref = rgb2lab(arr.astype(np.float64) / 255.0)
        name = os.path.basename(fn)

        # Part 1: K=64 三方法
        outs = {
            'kmeans_rgb': quantize_kmeans_sk(img, K),
            'pw_lab': pw_kmeans(img, K, use_lab=True, weight_power=0.5),
            'ce_kmeans': ce_kmeans(img, K),
        }
        for m, out in outs.items():
            oa = np.array(out)
            rows1.append({'image': name, 'method': m,
                          'psnr': round(psnr(arr, oa), 3),
                          'ssim': round(float(ssim_fn(arr, oa, channel_axis=2, data_range=255)), 4),
                          'dE00': round(float(deltaE_ciede2000(lab_ref,
                                  rgb2lab(oa.astype(np.float64) / 255.0)).mean()), 3)})

        # Part 2: 迭代搜索
        k, out, d = iterative_palette(img, TAU)
        oa = np.array(out)
        bio = io.BytesIO()
        out.convert('P', palette=Image.Palette.ADAPTIVE, colors=k).save(bio, format='PNG')
        rows2.append({'image': name, 'K_iter': k, 'dE00_iter': round(d, 3),
                      'bytes_iter': bio.tell(), 'psnr_iter': round(psnr(arr, oa), 2)})

        if (fi + 1) % 10 == 0 or fi == len(files) - 1:
            sys.stdout.write(f'[{fi+1}/{len(files)}] {name}: '
                             f'kmeans_dE00={rows1[-3]["dE00"]} ce_dE00={rows1[-1]["dE00"]} iter_K={k}\n')
            sys.stdout.flush()

    df1 = pd.DataFrame(rows1)
    df2 = pd.DataFrame(rows2)
    df1.to_csv(f'{_ROOT}/data/div2k_methods.csv', index=False)
    df2.to_csv(f'{_ROOT}/data/div2k_iterative.csv', index=False)

    print('\n=== DIV2K-100 K=64 comparison ===')
    for m in ['kmeans_rgb', 'pw_lab', 'ce_kmeans']:
        sub = df1[df1.method == m]
        print(f'{m:<12} dE00={sub.dE00.mean():.3f} PSNR={sub.psnr.mean():.2f} SSIM={sub.ssim.mean():.4f}')

    from scipy.stats import wilcoxon
    km_d = df1[df1.method == 'kmeans_rgb'].sort_values('image').dE00.values
    pw_d = df1[df1.method == 'pw_lab'].sort_values('image').dE00.values
    ce_d = df1[df1.method == 'ce_kmeans'].sort_values('image').dE00.values
    w1, p1 = wilcoxon(ce_d, km_d)
    w2, p2 = wilcoxon(ce_d, pw_d)
    print(f'Wilcoxon: CE vs K-means p={p1:.2e} (wins {(ce_d < km_d).sum()}/{len(ce_d)})')
    print(f'          CE vs PW-Lab  p={p2:.2e} (wins {(ce_d < pw_d).sum()}/{len(ce_d)})')

    print('\n=== DIV2K-100 iterative search (tau=%.1f) ===' % TAU)
    print(f'K distribution: {df2.K_iter.value_counts().sort_index().to_dict()}')
    print(f'meet rate: {(df2.dE00_iter <= TAU).mean():.3f}, avg bytes: {df2.bytes_iter.mean():.0f}')


if __name__ == '__main__':
    main()