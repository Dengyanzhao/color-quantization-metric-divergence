# -*- coding: utf-8 -*-
"""Beta sensitivity analysis: PW-Lab with beta in {0, 0.25, 0.5, 1.0} at K=64."""

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
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_kmeans_v3 import pw_kmeans, psnr
from skimage.metrics import structural_similarity as ssim_fn

KODAK_DIR = f'{_DATASETS}/kodak'
K = 64
betas = [0.0, 0.25, 0.5, 1.0]

def main():
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    acc = {b: {'psnr': [], 'ssim': [], 'd00': []} for b in betas}
    for fn in files:
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        arr = np.array(img)
        for b in betas:
            out = pw_kmeans(img, K, use_lab=True, weight_power=b)
            oa = np.array(out)
            acc[b]['psnr'].append(psnr(arr, oa))
            acc[b]['ssim'].append(float(ssim_fn(arr, oa, channel_axis=2, data_range=255)))
            acc[b]['d00'].append(float(deltaE_ciede2000(rgb2lab(arr/255.0), rgb2lab(oa/255.0)).mean()))
    print('=== Beta sensitivity (K=64, mean over 24 images) ===')
    print(f"{'beta':<6}{'PSNR':>8}{'SSIM':>8}{'dE00':>8}")
    for b in betas:
        print(f"{b:<6}{np.mean(acc[b]['psnr']):>8.2f}{np.mean(acc[b]['ssim']):>8.4f}{np.mean(acc[b]['d00']):>8.3f}")

if __name__ == '__main__':
    main()
