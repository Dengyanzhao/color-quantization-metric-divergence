# -*- coding: utf-8 -*-
"""Color quantization experiment: classic algorithms on Kodak test set."""

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
import time, csv, os, sys
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim_fn

KODAK_DIR = f'{_DATASETS}/kodak'
OUT_CSV = f'{_ROOT}/data/results.csv'
RNG = np.random.default_rng(42)

METHODS = ['mediancut', 'maxcoverage', 'fast', 'kmeans', 'som']
N_COLORS = [16, 32, 64, 128, 256]


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def quantize_pil(img, n, method):
    q = img.quantize(colors=n, method=method, dither=Image.Dither.NONE)
    return q.convert('RGB')


def quantize_kmeans(img, n):
    arr = np.array(img).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(42)
    sample = arr[rng.choice(len(arr), min(50000, len(arr)), replace=False)]
    km = KMeans(n_clusters=n, n_init=3, random_state=42, max_iter=100)
    km.fit(sample)
    labels = km.predict(arr)
    centers = km.cluster_centers_.round().astype(np.uint8)
    out = centers[labels].reshape(img.size[1], img.size[0], 3)
    return Image.fromarray(out)


def quantize_som(img, n):
    # Simplified 1D Kohonen SOM color quantizer
    arr = np.array(img).reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(42)
    W = rng.uniform(0, 255, (n, 3))
    pixels = arr[rng.choice(len(arr), min(30000, len(arr)), replace=False)]
    sigma = max(1.0, n / 8.0)
    n_iter = 200
    for t in range(n_iter):
        alpha = 0.3 * (1.0 - t / n_iter)
        idx = rng.integers(0, len(pixels))
        x = pixels[idx]
        d = np.linalg.norm(W - x, axis=1)
        bmu = int(np.argmin(d))
        dists = np.minimum(np.abs(np.arange(n) - bmu), n - np.abs(np.arange(n) - bmu))
        h = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))
        W += alpha * h[:, None] * (x - W)
    # assign labels in chunks to avoid huge allocation
    Wc = W.round().astype(np.uint8)
    labels = np.empty(len(arr), dtype=np.int32)
    chunk = 20000
    for i in range(0, len(arr), chunk):
        seg = arr[i:i + chunk]
        labels[i:i + chunk] = np.argmin(
            np.linalg.norm(seg[:, None, :] - W[None, :, :], axis=2), axis=1)
    out = Wc[labels].reshape(img.size[1], img.size[0], 3)
    return Image.fromarray(out)


def run_one(img_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    rows = []
    for n in N_COLORS:
        for m in METHODS:
            t0 = time.perf_counter()
            if m == 'kmeans':
                out = quantize_kmeans(img, n)
            elif m == 'som':
                out = quantize_som(img, n)
            elif m == 'mediancut':
                out = quantize_pil(img, n, Image.Quantize.MEDIANCUT)
            elif m == 'maxcoverage':
                out = quantize_pil(img, n, Image.Quantize.MAXCOVERAGE)
            else:
                out = quantize_pil(img, n, Image.Quantize.FASTOCTREE)
            dt = (time.perf_counter() - t0) * 1000.0
            out_arr = np.array(out)
            p = psnr(arr, out_arr)
            s = ssim_fn(arr, out_arr, channel_axis=2, data_range=255)
            # file size: palette PNG
            buf = out.convert('P', palette=Image.Palette.ADAPTIVE, colors=n)
            import io
            bio = io.BytesIO()
            buf.save(bio, format='PNG')
            nbytes = bio.tell()
            rows.append([os.path.basename(img_path), m, n,
                         round(p, 3), round(float(s), 4), round(dt, 1), nbytes])
            sys.stdout.write(f"{os.path.basename(img_path)} {m} n={n}: "
                             f"PSNR={p:.2f} SSIM={s:.4f} t={dt:.0f}ms\n")
            sys.stdout.flush()
    return rows


def main():
    files = sorted(os.listdir(KODAK_DIR))
    files = [f for f in files if f.endswith('.png')]
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'method', 'n_colors', 'psnr', 'ssim', 'time_ms', 'png_bytes'])
        for fn in files:
            rows = run_one(os.path.join(KODAK_DIR, fn))
            w.writerows(rows)
            f.flush()
    print('DONE. Results in', OUT_CSV)


if __name__ == '__main__':
    main()
