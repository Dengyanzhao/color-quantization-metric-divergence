# -*- coding: utf-8 -*-
"""Run all quantization methods on the full dataset, save per-method outputs.
Outputs:
  results/outputs/{method}/{image}.jpg   (quantized images)
  results/palettes.csv                   (real palette colors per image/method)
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
import sys, os, csv, time
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ce_kmeans import ce_kmeans

K = 64
SRC_DIR = f'{_DATASETS}/kodak'
OUT_DIR = f'{_ROOT}/results'
JPEG_Q = 95

METHODS = ['original', 'ce_kmeans', 'mediancut', 'fastoctree', 'kmeans', 'som']
METHOD_LABEL = {'original': 'Original (24-bit)', 'ce_kmeans': 'CE-KMeans (Ours)',
                'mediancut': 'Median Cut', 'fastoctree': 'Fast Octree',
                'kmeans': 'K-means (RGB)', 'som': 'SOM'}


def quantize_mediancut(img, k):
    return img.quantize(colors=k, method=Image.Quantize.MEDIANCUT,
                        dither=Image.Dither.NONE).convert('RGB')


def quantize_fast(img, k):
    return img.quantize(colors=k, method=Image.Quantize.FASTOCTREE,
                        dither=Image.Dither.NONE).convert('RGB')


def quantize_kmeans(img, k):
    arr = np.array(img).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(42)
    km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
    km.fit(arr[rng.choice(len(arr), min(50000, len(arr)), replace=False)])
    out = km.cluster_centers_.round().astype(np.uint8)[km.predict(arr)]
    return Image.fromarray(out.reshape(img.size[1], img.size[0], 3))


def quantize_som(img, k):
    arr = np.array(img).reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(42)
    W = rng.uniform(0, 255, (k, 3))
    pixels = arr[rng.choice(len(arr), min(30000, len(arr)), replace=False)]
    sigma = max(1.0, k / 8.0)
    for t in range(200):
        alpha = 0.3 * (1.0 - t / 200)
        idx = rng.integers(0, len(pixels))
        x = pixels[idx]
        d = np.linalg.norm(W - x, axis=1)
        bmu = int(np.argmin(d))
        dists = np.minimum(np.abs(np.arange(k) - bmu), k - np.abs(np.arange(k) - bmu))
        W += alpha * np.exp(-(dists ** 2) / (2.0 * sigma ** 2))[:, None] * (x - W)
    Wc = W.round().astype(np.uint8)
    labels = np.argmin(np.linalg.norm(arr[:, None, :] - W[None, :, :], axis=2), axis=1)
    return Image.fromarray(Wc[labels].reshape(img.size[1], img.size[0], 3))


def real_palette(out_rgb):
    """Real palette colors + pixel counts from the actual output image."""
    arr = np.array(out_rgb).reshape(-1, 3)
    colors, counts = np.unique(arr, axis=0, return_counts=True)
    order = np.argsort(-counts)
    return colors[order].astype(np.uint8), counts[order]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for m in METHODS:
        os.makedirs(os.path.join(OUT_DIR, 'outputs', m), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'mosaics'), exist_ok=True)

    files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith('.png')])
    with open(os.path.join(OUT_DIR, 'palettes.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'method', 'n_real_colors', 'palette_rgb', 'counts'])
        for fn in files:
            img = Image.open(os.path.join(SRC_DIR, fn)).convert('RGB')
            variants = {'original': img,
                        'ce_kmeans': ce_kmeans(img, K),
                        'mediancut': quantize_mediancut(img, K),
                        'fastoctree': quantize_fast(img, K),
                        'kmeans': quantize_kmeans(img, K),
                        'som': quantize_som(img, K)}
            base = fn.replace('.png', '')
            for m, out in variants.items():
                out_path = os.path.join(OUT_DIR, 'outputs', m, f'{base}.jpg')
                out.save(out_path, 'JPEG', quality=JPEG_Q)
                colors, counts = real_palette(out)
                w.writerow([base, m, len(colors),
                            colors.flatten().tolist(), counts.tolist()])
                sys.stdout.write(f'{base} [{m:<12}] {len(colors)} colors\n')
                sys.stdout.flush()
            f.flush()
    print('DONE. Outputs in', OUT_DIR)


if __name__ == '__main__':
    main()