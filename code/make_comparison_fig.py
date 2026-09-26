# -*- coding: utf-8 -*-
"""Generate paper-style method comparison mosaic:
each cell = quantized image + palette color bar (proportional).
Methods: Original / CE-KMeans (Ours) / Median Cut / Fast Octree / K-means / SOM
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
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ce_kmeans import ce_kmeans
from sklearn.cluster import KMeans

K = 64
IMG_PATH = f'{_DATASETS}/kodak/kodim01.png'
OUT = f'{_ROOT}/figures/fig12_methods_comparison.png'


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
    """Simplified 1D Kohonen SOM (from experiment.py)."""
    arr = np.array(img).reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(42)
    W = rng.uniform(0, 255, (k, 3))
    pixels = arr[rng.choice(len(arr), min(30000, len(arr)), replace=False)]
    sigma = max(1.0, k / 8.0)
    n_iter = 200
    for t in range(n_iter):
        alpha = 0.3 * (1.0 - t / n_iter)
        idx = rng.integers(0, len(pixels))
        x = pixels[idx]
        d = np.linalg.norm(W - x, axis=1)
        bmu = int(np.argmin(d))
        dists = np.minimum(np.abs(np.arange(k) - bmu), k - np.abs(np.arange(k) - bmu))
        h = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))
        W += alpha * h[:, None] * (x - W)
    Wc = W.round().astype(np.uint8)
    labels = np.argmin(np.linalg.norm(arr[:, None, :] - W[None, :, :], axis=2), axis=1)
    return Image.fromarray(Wc[labels].reshape(img.size[1], img.size[0], 3))


def palette_bar(img, k):
    """Extract palette colors + proportional counts as (colors, counts)."""
    q = img.convert('P', palette=Image.Palette.ADAPTIVE, colors=k)
    pal = np.array(q.getpalette()[:k * 3]).reshape(-1, 3).astype(np.uint8)
    idx = np.array(q).ravel()
    max_idx = int(idx.max())
    pal = pal[:max_idx + 1]
    counts = np.bincount(idx, minlength=max_idx + 1).astype(np.float64)
    # 只保留用到的颜色
    mask = counts > 0
    return pal[mask], counts[mask]


def main():
    img = Image.open(IMG_PATH).convert('RGB')
    methods = [
        ('Original', img, 'Original (24-bit)'),
        ('ours', ce_kmeans(img, K), 'CE-KMeans (Ours)'),
        ('mediancut', quantize_mediancut(img, K), 'Median Cut'),
        ('fast', quantize_fast(img, K), 'Fast Octree'),
        ('kmeans', quantize_kmeans(img, K), 'K-means (RGB)'),
        ('som', quantize_som(img, K), 'SOM'),
    ]

    fig = plt.figure(figsize=(11, 15))
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.12,
                  height_ratios=[5, 5, 5])
    for i, (name, out, title) in enumerate(methods):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.imshow(np.array(out))
        ax.set_title(title, fontsize=11, pad=2)
        ax.axis('off')
        # 下方颜色条
        colors, counts = palette_bar(out, K)
        order = np.argsort(-counts)
        colors = colors[order]
        fracs = counts[order] / counts.sum()
        axb = fig.add_axes([0, 0, 0, 0])  # placeholder, removed below
        axb.remove()
        # 用 inset 画颜色条：底部 8% 高度
        axins = ax.inset_axes([0.02, -0.06, 0.96, 0.045])
        left = 0.0
        for c, f in zip(colors, fracs):
            axins.barh(0, f, left=left, color=np.array(c) / 255.0,
                       edgecolor='none', height=1.0)
            left += f
        axins.set_xlim(0, 1)
        axins.set_ylim(0, 1)
        axins.axis('off')
    fig.suptitle('Color Quantization Comparison on kodim01 (K=64)', fontsize=14, y=0.995)
    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close()
    print('saved:', OUT)


if __name__ == '__main__':
    main()