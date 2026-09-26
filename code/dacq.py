# -*- coding: utf-8 -*-
"""DACQ: Dither-Aware Color Quantization framework.
Stage 1: CE-KMeans initial palette.
Stage 2: luminance sort.
Stage 3: dither-aware palette refinement (noise-annealed perceptual assignment).
Stage 4: final Floyd-Steinberg dithering with refined palette.
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
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim_fn
import sys, os, csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ce_kmeans import ce_kmeans, psnr, assign_ce


def luminance_sort(palette):
    Y = 0.299 * palette[:, 0] + 0.587 * palette[:, 1] + 0.114 * palette[:, 2]
    return palette[np.argsort(Y)]


def dither_aware_refine(img, palette, n_iter=6, sigma0=16.0):
    """Stage 3: refine palette to be robust to dithering noise."""
    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    pixels = arr.reshape(-1, 3)
    pal = palette.astype(np.float64)
    for it in range(n_iter):
        sigma = sigma0 * (1.0 - it / n_iter) + 1.0
        rng = np.random.default_rng(42 + it)
        noise = rng.normal(0, sigma, (h, w, 3))
        noisy = np.clip(arr + noise, 0, 255)
        noisy_lab = rgb2lab(noisy / 255.0).reshape(-1, 3).astype(np.float32)
        pal_lab = rgb2lab(pal[None, :, :] / 255.0)[0].astype(np.float32)
        labels = assign_ce(noisy_lab, pal_lab, chunk=15000)
        new_pal = pal.copy()
        for k in range(len(pal)):
            m = (labels == k)
            if m.sum() > 0:
                new_pal[k] = pixels[m].mean(axis=0)
        if np.abs(new_pal - pal).max() < 0.5:
            break
        pal = new_pal
    return np.clip(pal, 0, 255).astype(np.uint8)


def make_palette_image(palette):
    pal_img = Image.new('P', (1, 1))
    full = palette.flatten().tolist() + [0] * (256 - len(palette)) * 3
    pal_img.putpalette(full)
    return pal_img


def main():
    KODAK_DIR = f'{_DATASETS}/kodak'
    K = 64
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    rows = []
    for fn in files:
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        arr = np.array(img)
        out_ce = ce_kmeans(img, K)
        q = out_ce.convert('P', palette=Image.Palette.ADAPTIVE, colors=K)
        pal = np.array(q.getpalette()[:K * 3]).reshape(-1, 3).astype(np.uint8)
        pal_sorted = luminance_sort(pal)
        pal_refined = dither_aware_refine(img, pal_sorted)
        pal_img_plain = make_palette_image(pal_sorted)
        pal_img_refined = make_palette_image(pal_refined)
        variants = {
            'CE no-dither': out_ce,
            'CE + plain FS': img.quantize(palette=pal_img_plain,
                                          dither=Image.Dither.FLOYDSTEINBERG).convert('RGB'),
            'DACQ (refined+FS)': img.quantize(palette=pal_img_refined,
                                              dither=Image.Dither.FLOYDSTEINBERG).convert('RGB'),
        }
        line = fn
        for name, out in variants.items():
            oa = np.array(out)
            p = psnr(arr, oa)
            s = float(ssim_fn(arr, oa, channel_axis=2, data_range=255))
            d = float(deltaE_ciede2000(rgb2lab(arr.astype(np.float64) / 255.0),
                                       rgb2lab(oa.astype(np.float64) / 255.0)).mean())
            rows.append([fn, name, round(p, 2), round(s, 4), round(d, 3)])
            line += f' | {name}: PSNR={p:.1f} dE00={d:.2f}'
        sys.stdout.write(line + '\n')
        sys.stdout.flush()
    with open(f'{_ROOT}/data/dacq_results.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['image','method','psnr','ssim','dE00']); w.writerows(rows)
    print('\n=== DACQ (K=64, Kodak 24) ===')
    import pandas as pd
    df = pd.read_csv(f'{_ROOT}/data/dacq_results.csv')
    for m in df.method.unique():
        sub = df[df.method == m]
        print(f'{m:<20} PSNR={sub.psnr.mean():.1f} dE00={sub.dE00.mean():.3f} SSIM={sub.ssim.mean():.4f}')


if __name__ == '__main__':
    main()