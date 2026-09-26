# -*- coding: utf-8 -*-
"""Viewing-distance evaluation: does CE-KMeans palette improve dithered quality?
Compares (all K=64, Kodak 24, viewing distance = 4x downsample):
  - K-means palette + FS  vs  CE palette + FS   (palette quality transfer)
  - CE no-dither         vs  CE + FS            (dithering value)
  - refined (gentle) + FS                      (DACQ v2: sigma 8->1)
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

import numpy as np, os, sys
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_ciede2000
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ce_kmeans import ce_kmeans, assign_ce
from dacq import luminance_sort, make_palette_image

KODAK_DIR = f'{_DATASETS}/kodak'
K = 64


def dither_aware_refine_v2(img, palette, n_iter=4, sigma0=8.0):
    """Gentle refinement: small noise (closer to actual FS residual)."""
    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    pixels = arr.reshape(-1, 3)
    pal = palette.astype(np.float64)
    for it in range(n_iter):
        sigma = sigma0 * (1.0 - it / n_iter) + 0.5
        rng = np.random.default_rng(100 + it)
        noisy = np.clip(arr + rng.normal(0, sigma, (h, w, 3)), 0, 255)
        noisy_lab = rgb2lab(noisy / 255.0).reshape(-1, 3).astype(np.float32)
        pal_lab = rgb2lab(pal[None, :, :] / 255.0)[0].astype(np.float32)
        labels = assign_ce(noisy_lab, pal_lab, chunk=15000)
        new_pal = pal.copy()
        for k in range(len(pal)):
            m = (labels == k)
            if m.sum() > 0:
                new_pal[k] = pixels[m].mean(axis=0)
        if np.abs(new_pal - pal).max() < 0.3:
            break
        pal = new_pal
    return np.clip(pal, 0, 255).astype(np.uint8)


def view_dE00(ref, out, scale=4):
    h, w = ref.shape[:2]
    nh, nw = h // scale, w // scale
    rs = np.array(Image.fromarray(ref).resize((nw, nh), Image.BILINEAR))
    os_ = np.array(Image.fromarray(out).resize((nw, nh), Image.BILINEAR))
    return float(deltaE_ciede2000(rgb2lab(rs.astype(np.float64)/255.0),
                                  rgb2lab(os_.astype(np.float64)/255.0)).mean())


def get_palette(img, method):
    if method == 'kmeans':
        arr = np.array(img).reshape(-1, 3).astype(np.float32)
        km = KMeans(n_clusters=K, n_init=3, random_state=42, max_iter=100)
        km.fit(arr[np.random.default_rng(42).choice(len(arr), 50000, replace=False)])
        return km.cluster_centers_.round().astype(np.uint8)
    else:  # ce_kmeans
        out = ce_kmeans(img, K)
        q = out.convert('P', palette=Image.Palette.ADAPTIVE, colors=K)
        return np.array(q.getpalette()[:K*3]).reshape(-1, 3).astype(np.uint8)


def main():
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    rows = []
    for fn in files:
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        arr = np.array(img)

        pal_km = luminance_sort(get_palette(img, 'kmeans'))
        pal_ce = luminance_sort(get_palette(img, 'ce_kmeans'))
        pal_ref = dither_aware_refine_v2(img, pal_ce)

        variants = {
            'CE no-dither': ce_kmeans(img, K),
            'Kmeans pal + FS': img.quantize(palette=make_palette_image(pal_km),
                                            dither=Image.Dither.FLOYDSTEINBERG).convert('RGB'),
            'CE pal + FS': img.quantize(palette=make_palette_image(pal_ce),
                                        dither=Image.Dither.FLOYDSTEINBERG).convert('RGB'),
            'CE pal + refined + FS': img.quantize(palette=make_palette_image(pal_ref),
                                                  dither=Image.Dither.FLOYDSTEINBERG).convert('RGB'),
        }
        v = {n: view_dE00(arr, np.array(o)) for n, o in variants.items()}
        p = {n: float(deltaE_ciede2000(rgb2lab(arr.astype(np.float64)/255.0),
                                       rgb2lab(np.array(o).astype(np.float64)/255.0)).mean())
             for n, o in variants.items()}
        row = {'image': fn}
        for n in variants:
            row[f'{n}_view'] = round(v[n], 3)
            row[f'{n}_pix'] = round(p[n], 3)
        rows.append(row)
        sys.stdout.write(f"{fn}: view[nd={v['CE no-dither']:.2f} kmfs={v['Kmeans pal + FS']:.2f} "
                         f"cefs={v['CE pal + FS']:.2f} ref={v['CE pal + refined + FS']:.2f}]\n")
        sys.stdout.flush()

    df = pd.DataFrame(rows)
    df.to_csv(f'{_ROOT}/data/viewing_distance_eval.csv', index=False)
    print('\n=== 观看距离 dE00（下采样 4x，均值）===')
    for n in variants:
        print(f'{n:<24} view={df[f"{n}_view"].mean():.3f}  pix={df[f"{n}_pix"].mean():.3f}')
    print(f'\nCE+FS vs Kmeans+FS (view): {(df["CE pal + FS_view"] < df["Kmeans pal + FS_view"]).sum()}/24 胜')
    print(f'CE+FS vs no-dither (view): {(df["CE pal + FS_view"] < df["CE no-dither_view"]).sum()}/24 胜')
    print(f'refined vs plain CE+FS (view): {(df["CE pal + refined + FS_view"] < df["CE pal + FS_view"]).sum()}/24 胜')


if __name__ == '__main__':
    main()