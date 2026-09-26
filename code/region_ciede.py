# -*- coding: utf-8 -*-
"""Region-stratified CIEDE2000: flat vs edge regions, PW-Lab vs K-means (K=64)."""

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
from skimage.filters import sobel
from sklearn.cluster import KMeans
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_kmeans_v3 import pw_kmeans

KODAK_DIR = f'{_DATASETS}/kodak'
K = 64

def main():
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    acc = {'km_flat': [], 'km_edge': [], 'pw_flat': [], 'pw_edge': []}
    for fn in files:
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        arr = np.array(img)
        gray = np.array(img.convert('L')).astype(np.float64)
        grad = sobel(gray)
        flat = grad < np.percentile(grad, 50)
        edge = grad >= np.percentile(grad, 90)
        # K-means baseline
        km = KMeans(n_clusters=K, n_init=3, random_state=42, max_iter=100)
        p = arr.reshape(-1, 3).astype(np.float32)
        km.fit(p[np.random.default_rng(42).choice(len(p), 50000, replace=False)])
        out_km = km.cluster_centers_.round().astype(np.uint8)[km.predict(p)].reshape(arr.shape)
        # PW-Lab
        out_pw = np.array(pw_kmeans(img, K, use_lab=True, weight_power=0.5))
        for tag, out in [('km', out_km), ('pw', out_pw)]:
            d = deltaE_ciede2000(rgb2lab(arr/255.0), rgb2lab(out/255.0))
            acc[f'{tag}_flat'].append(float(d[flat].mean()))
            acc[f'{tag}_edge'].append(float(d[edge].mean()))
    print('=== Region-stratified CIEDE2000 (K=64, mean over 24) ===')
    print(f"{'region':<8}{'K-means':>10}{'PW-Lab':>10}{'improve':>10}")
    for region in ['flat', 'edge']:
        a = np.mean(acc[f'km_{region}'])
        b = np.mean(acc[f'pw_{region}'])
        print(f"{region:<8}{a:>10.3f}{b:>10.3f}{(1-b/a)*100:>9.1f}%")

if __name__ == '__main__':
    main()
