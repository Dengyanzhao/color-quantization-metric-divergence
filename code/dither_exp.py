# -*- coding: utf-8 -*-
"""LEGACY / SUPERSEDED: early dithering experiment. Its output CSV had a bug
(the "no dither" row was computed on a palette-index image rather than the RGB
quantized image, and "ordered dither" duplicated it), so the file was moved to
paper/data/backup_old/dither_results_legacy_buggy.csv. The dithering results
reported in the paper come from the viewing-distance protocol (view_eval.py ->
viewing_distance_eval.csv) and from dacq.py -> dacq_results.csv, both regenerated
with the complete CIEDE2000 implementation. Do NOT cite this script's output.
Dithering + palette optimization for CE-KMeans and variants.
Implements:
1) Floyd-Steinberg error diffusion dithering
2) Palette sorting by luminance (reduces dither artifacts)
3) Dither-aware palette refinement: fine-tune palette to minimize
   post-dither total error (one round of correction)"""

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
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim_fn
import sys, os, io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ce_kmeans import ce_kmeans, psnr

KODAK_DIR = f'{_DATASETS}/kodak'
K = 64


def sort_palette_by_luminance(centers):
    """Sort palette by Y (luminance from RGB)."""
    Y = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
    return centers[np.argsort(Y)]


def floyd_steinberg(img, palette):
    """Floyd-Steinberg error diffusion dither with a given palette.
    palette: (K, 3) uint8. Returns PIL Image (RGB) and mean dE00."""
    palette = palette.astype(np.int32)
    arr = np.array(img).astype(np.int32)
    h, w = arr.shape[:2]
    out = np.zeros_like(arr)
    for y in range(h):
        for x in range(w):
            old = arr[y, x].copy()
            # find nearest palette color
            dist = np.sum((old.astype(np.float64) - palette.astype(np.float64)) ** 2, axis=1)
            idx = np.argmin(dist)
            new = palette[idx]
            out[y, x] = new
            err = old - new
            # diffuse error to neighbors
            if x + 1 < w:
                arr[y, x + 1] += err * 7 // 16
            if y + 1 < h:
                if x > 0:
                    arr[y + 1, x - 1] += err * 3 // 16
                arr[y + 1, x] += err * 5 // 16
                if x + 1 < w:
                    arr[y + 1, x + 1] += err * 1 // 16
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def ordered_dither(img, palette):
    """Bayer 4x4 ordered dither (no worm artifacts, regular pattern)."""
    # Bayer matrix 4x4
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                      [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0 - 0.5
    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    palette_f = palette.astype(np.float64)
    out = np.zeros_like(arr, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            # add bayer noise
            noisy = arr[y, x] + bayer[y % 4, x % 4] * 255 * 0.15
            noisy = np.clip(noisy, 0, 255)
            dist = np.sum((noisy[None, :] - palette_f) ** 2, axis=1)
            out[y, x] = palette[np.argmin(dist)]
    return Image.fromarray(out)


def refine_palette_for_dither(img, palette, n_iter=5):
    """One round of palette refinement: minimize dither+measure error."""
    # Simple: adjust palette centroids toward pixels that are commonly dithered
    # to far-away colors. This is a heuristic approximation.
    arr = np.array(img).astype(np.float64)
    palette_f = palette.astype(np.float64).copy()
    for _ in range(n_iter):
        # assign with small noise (simulate dither)
        noise = np.random.default_rng(42).uniform(-8, 8, arr.shape)
        noisy = np.clip(arr + noise, 0, 255)
        labels = np.argmin(np.sum((noisy[:, :, None, :] - palette_f[None, None, :, :]) ** 2, axis=3), axis=2)
        for k in range(len(palette)):
            mask = labels == k
            if mask.sum() > 0:
                palette_f[k] = arr[mask].mean(axis=0)
    return palette_f.astype(np.uint8)


def main():
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    rows = []
    for fn in files:
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        arr = np.array(img)

        # 1. CE-KMeans palette (unsorted)
        out_ce = ce_kmeans(img, K)
        q = out_ce.convert('P', palette=Image.Palette.ADAPTIVE, colors=K)
        pal = np.array(q.getpalette()[:K*3]).reshape(-1, 3).astype(np.uint8)

        # 2. Sorted palette
        pal_sorted = sort_palette_by_luminance(pal)

        # 3. FS dither with sorted palette
        out_fs = floyd_steinberg(img, pal_sorted)
        # 4. Ordered dither with sorted palette
        out_od = ordered_dither(img, pal_sorted)
        # 5. Refined palette + FS dither
        pal_refined = refine_palette_for_dither(img, pal_sorted)
        out_rf = floyd_steinberg(img, pal_refined)

        # 6. Baseline: CE-KMeans no dither
        out_nd = out_ce

        variants = {
            'CE-KMeans (no dither)': out_nd,
            'CE + FS dither': out_fs,
            'CE + ordered dither': out_od,
            'CE + refined + FS': out_rf,
        }
        line = fn
        for name, out in variants.items():
            oa = np.array(out)
            p = psnr(arr, oa)
            s = float(ssim_fn(arr, oa, channel_axis=2, data_range=255))
            d = float(deltaE_ciede2000(rgb2lab(arr.astype(np.float64)/255.0),
                                       rgb2lab(oa.astype(np.float64)/255.0)).mean())
            rows.append({'image': fn, 'method': name, 'psnr': round(p,2),
                         'ssim': round(s,4), 'dE00': round(d,3)})
            line += f' | {name}: PSNR={p:.1f} dE00={d:.2f}'
        sys.stdout.write(line + '\n')
        sys.stdout.flush()

    # summary
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f'{_ROOT}/data/dither_results.csv', index=False)
    print('\n=== Dither comparison (K=%d, mean over 24 Kodak) ===' % K)
    for m in df.method.unique():
        sub = df[df.method == m]
        print(f'{m:<25} PSNR={sub.psnr.mean():.1f} dE00={sub.dE00.mean():.3f} SSIM={sub.ssim.mean():.4f}')


if __name__ == '__main__':
    main()