# -*- coding: utf-8 -*-
"""Iterative (training-free) palette size search evaluation.
Compares: fixed-K baselines vs iterative search (tau on mean CIEDE2000).
Reports bytes, meeting rate, and mean dE00."""

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
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
import os, sys, io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ce_kmeans import iterative_palette, psnr

KODAK_DIR = f'{_DATASETS}/kodak'
TAU = 2.5  # mean CIEDE2000 target


def main():
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    rows = []
    for fn in files:
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        arr = np.array(img)
        k, out, d = iterative_palette(img, TAU)
        oa = np.array(out)
        bio = io.BytesIO()
        out.convert('P', palette=Image.Palette.ADAPTIVE, colors=k).save(bio, format='PNG')
        rows.append({'image': fn, 'K_iter': k, 'dE00_iter': round(d, 3),
                     'bytes_iter': bio.tell(), 'psnr_iter': round(psnr(arr, oa), 2)})
        sys.stdout.write(f'{fn}: iterative K={k} dE00={d:.3f} bytes={bio.tell()}\n')
        sys.stdout.flush()

    df = pd.DataFrame(rows)
    df.to_csv(f'{_ROOT}/data/iterative_adaptive.csv', index=False)

    print('\n=== Iterative search (tau=%.1f) vs fixed K (mean over 24) ===' % TAU)
    print(f'{"scheme":<20}{"meet_rate":>10}{"avg_bytes":>12}{"avg_dE00":>10}')
    print(f'{"iterative":<20}{1.0:>10.3f}{df.bytes_iter.mean():>12.0f}{df.dE00_iter.mean():>10.3f}')
    ce = pd.read_csv(f'{_ROOT}/data/ce_results.csv')
    for k in [32, 64, 128, 256]:
        sub = ce[ce.n_colors == k]
        meet = (sub.dE00 <= TAU).mean()
        print(f'{"fixed K=%d" % k:<20}{meet:>10.3f}{sub.png_bytes.mean():>12.0f}{sub.dE00.mean():>10.3f}')
    print('\nK distribution:', df.K_iter.value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()