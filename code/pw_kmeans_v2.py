# -*- coding: utf-8 -*-
"""PW-KMeans v2: Perceptual-Weighted K-means in CIELAB (subsample training + full-image assign)."""

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
from skimage.color import rgb2lab, lab2rgb
from skimage.filters import sobel
import time, csv, os, sys, io

RNG = np.random.default_rng(42)
KODAK_DIR = f'{_DATASETS}/kodak'
OUT_CSV = f'{_ROOT}/data/pw_results.csv'
N_COLORS = [16, 32, 64, 128, 256]
SAMPLE = 50000
MAX_ITER = 30


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def assign_chunked(pixels, centers, chunk=30000):
    n = len(pixels)
    labels = np.empty(n, dtype=np.int32)
    for i in range(0, n, chunk):
        seg = pixels[i:i + chunk]
        labels[i:i + chunk] = np.argmin(
            np.linalg.norm(seg[:, None, :] - centers[None, :, :], axis=2), axis=1)
    return labels


def kmeanspp_sub(pixels, k, rng):
    """K-means++ init on subsampled pixels (fast)."""
    n = len(pixels)
    centers = pixels[rng.choice(n, 1)]
    for _ in range(1, k):
        # distance to nearest center (batch)
        d = np.full(n, np.inf)
        for i in range(0, n, 20000):
            seg = pixels[i:i + 20000]
            d[i:i + 20000] = np.min(
                np.linalg.norm(seg[:, None, :] - centers[None, :, :], axis=2), axis=1)
        prob = d ** 2
        prob = prob / prob.sum()
        idx = rng.choice(n, 1, p=prob)
        centers = np.vstack([centers, pixels[idx]])
    return centers


def pw_kmeans(img, n_colors, weight_power=0.5):
    """Perceptual-Weighted K-means: CIELAB space + gradient-weighted centroids."""
    arr = np.array(img).astype(np.float64) / 255.0
    lab = rgb2lab(arr)
    h, w, _ = lab.shape
    pixels = lab.reshape(-1, 3).astype(np.float32)

    # gradient weights
    gray = np.array(img.convert('L')).astype(np.float64)
    grad = sobel(gray)
    w_flat = (1.0 + weight_power * (grad / (grad.max() + 1e-8))).ravel().astype(np.float32)

    # subsample for training
    idx = RNG.choice(len(pixels), min(SAMPLE, len(pixels)), replace=False)
    tr_pix = pixels[idx]
    tr_w = w_flat[idx]

    centers = kmeanspp_sub(tr_pix, n_colors, RNG).astype(np.float32)

    for _ in range(MAX_ITER):
        labels = assign_chunked(tr_pix, centers, chunk=10000)
        new_centers = np.zeros_like(centers)
        for k in range(n_colors):
            m = (labels == k)
            s = m.sum()
            if s > 0:
                new_centers[k] = np.average(tr_pix[m], axis=0, weights=tr_w[m])
            else:
                new_centers[k] = centers[k]
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    centers_rgb = lab2rgb(centers[None, ...])[0].clip(0, 1)
    centers_rgb = (centers_rgb * 255).astype(np.uint8)

    labels = assign_chunked(pixels, centers)
    out = centers_rgb[labels].reshape(h, w, 3)
    return Image.fromarray(out)


def run_one(img_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    rows = []
    for n in N_COLORS:
        t0 = time.perf_counter()
        out = pw_kmeans(img, n)
        dt = (time.perf_counter() - t0) * 1000.0
        out_arr = np.array(out)
        p = psnr(arr, out_arr)
        from skimage.metrics import structural_similarity as ssim_fn
        s = ssim_fn(arr, out_arr, channel_axis=2, data_range=255)
        buf = out.convert('P', palette=Image.Palette.ADAPTIVE, colors=n)
        bio = io.BytesIO()
        buf.save(bio, format='PNG')
        nbytes = bio.tell()
        rows.append([os.path.basename(img_path), 'pw_kmeans', n,
                     round(p, 3), round(float(s), 4), round(dt, 1), nbytes])
        sys.stdout.write(f"{os.path.basename(img_path)} pw_kmeans n={n}: "
                         f"PSNR={p:.2f} SSIM={s:.4f} t={dt:.0f}ms\n")
        sys.stdout.flush()
    return rows


def main():
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'method', 'n_colors', 'psnr', 'ssim', 'time_ms', 'png_bytes'])
        for fn in files:
            rows = run_one(os.path.join(KODAK_DIR, fn))
            w.writerows(rows)
            f.flush()
    print('DONE. PW-KMeans results in', OUT_CSV)


if __name__ == '__main__':
    main()