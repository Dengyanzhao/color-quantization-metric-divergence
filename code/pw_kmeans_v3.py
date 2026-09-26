# -*- coding: utf-8 -*-
"""PW-KMeans v3: Perceptual-Weighted K-means in CIELAB.
Init from median-cut palette (fast), then weighted refinement in Lab space.
Also runs RGB-space weighted variant for ablation."""

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
MAX_ITER = 20


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def assign_chunked(pixels, centers, chunk=20000):
    n = len(pixels)
    labels = np.empty(n, dtype=np.int32)
    for i in range(0, n, chunk):
        seg = pixels[i:i + chunk]
        labels[i:i + chunk] = np.argmin(
            np.linalg.norm(seg[:, None, :] - centers[None, :, :], axis=2), axis=1)
    return labels


def weighted_kmeans(pixels, weights, centers, max_iter):
    """Weighted K-means refinement. pixels/weights float32, centers float32."""
    for _ in range(max_iter):
        labels = assign_chunked(pixels, centers, chunk=10000)
        new_centers = np.zeros_like(centers)
        for k in range(len(centers)):
            m = (labels == k)
            s = m.sum()
            if s > 0:
                new_centers[k] = np.average(pixels[m], axis=0, weights=weights[m])
            else:
                new_centers[k] = centers[k]
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers
    return centers


def pw_kmeans(img, n_colors, use_lab=True, weight_power=0.5):
    """
    Perceptual-Weighted K-means.
    use_lab=True  -> CIELAB space (perceptual)
    use_lab=False -> RGB space (ablation)
    weight_power=0 -> no gradient weighting (ablation)
    """
    arr = np.array(img).astype(np.float64) / 255.0
    if use_lab:
        work = rgb2lab(arr)
        h, w, _ = work.shape
        pixels = work.reshape(-1, 3).astype(np.float32)
    else:
        h, w, _ = arr.shape
        pixels = arr.reshape(-1, 3).astype(np.float32) * 255.0

    # gradient weights
    gray = np.array(img.convert('L')).astype(np.float64)
    grad = sobel(gray)
    w_flat = (1.0 + weight_power * (grad / (grad.max() + 1e-8))).ravel().astype(np.float32)

    # init centers from median-cut palette (RGB), convert to working space
    mc = img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    centers_rgb = np.array(mc.getpalette()[:n_colors * 3]).reshape(-1, 3).astype(np.float64)
    if use_lab:
        centers = rgb2lab(centers_rgb[None, :, :] / 255.0)[0].astype(np.float32)
    else:
        centers = centers_rgb.astype(np.float32)

    # subsample for training
    idx = RNG.choice(len(pixels), min(SAMPLE, len(pixels)), replace=False)
    tr_pix = pixels[idx]
    tr_w = w_flat[idx]

    centers = weighted_kmeans(tr_pix, tr_w, centers, MAX_ITER)

    # full-image assignment
    labels = assign_chunked(pixels, centers)
    if use_lab:
        centers_out = lab2rgb(centers[None, ...])[0].clip(0, 1)
    else:
        centers_out = centers / 255.0
    centers_u8 = (centers_out * 255).astype(np.uint8)
    out = centers_u8[labels].reshape(h, w, 3)
    return Image.fromarray(out)


def run_one(img_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    rows = []
    variants = [
        ('pw_lab', True, 0.5),      # proposed: CIELAB + gradient weight
        ('pw_lab_now', True, 0.0),  # ablation: CIELAB only
        ('pw_rgb_w', False, 0.5),   # ablation: RGB + gradient weight
    ]
    for n in N_COLORS:
        for name, use_lab, wp in variants:
            t0 = time.perf_counter()
            out = pw_kmeans(img, n, use_lab=use_lab, weight_power=wp)
            dt = (time.perf_counter() - t0) * 1000.0
            out_arr = np.array(out)
            p = psnr(arr, out_arr)
            from skimage.metrics import structural_similarity as ssim_fn
            s = ssim_fn(arr, out_arr, channel_axis=2, data_range=255)
            buf = out.convert('P', palette=Image.Palette.ADAPTIVE, colors=n)
            bio = io.BytesIO()
            buf.save(bio, format='PNG')
            nbytes = bio.tell()
            rows.append([os.path.basename(img_path), name, n,
                         round(p, 3), round(float(s), 4), round(dt, 1), nbytes])
            sys.stdout.write(f"{os.path.basename(img_path)} {name} n={n}: "
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
    print('DONE. PW results in', OUT_CSV)


if __name__ == '__main__':
    main()