# -*- coding: utf-8 -*-
"""Reference-style mosaic v5 — aligned with venetian_lagoon reference.
- Top title: "#01 kodim01  K=64" (clean, no global dE00)
- Cell titles: "Ours  dE00=1.779", SOM gets "(63 colors)"
- dE00 in title (right side), NOT overlaid on image
- Palette grid with per-block % labels; NO color-count text
- Original cell: plain title, no dE00, no color count
- Ours green frame, uniform fonts (regular weight)
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
import sys, os, csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from skimage.color import rgb2lab, deltaE_ciede2000

# 统一常规字重（与参考图一致）
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'figure.facecolor': 'white',
})

RESULT_DIR = f'{_ROOT}/results'
K = 64
TOP = [('original', False), ('ce_kmeans', True), ('mediancut', False), ('fastoctree', False)]
BOT = [('original', False), ('kmeans', False), ('som', False), ('maxcoverage', False)]
LABEL = {'original': 'Original', 'ce_kmeans': 'Ours', 'mediancut': 'MedianCut',
         'fastoctree': 'Octree', 'kmeans': 'Kmeans', 'som': 'SOM',
         'maxcoverage': 'MaxCoverage'}
GREEN = '#00A651'


def load_palettes():
    import csv
    csv.field_size_limit(2**30)
    pal = {}
    with open(os.path.join(RESULT_DIR, 'palettes.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            colors = np.array(json.loads(row['palette_rgb']), dtype=np.uint8).reshape(-1, 3)
            counts = np.array(json.loads(row['counts']), dtype=np.float64)
            pal.setdefault(row['image'], {})[row['method']] = (colors, counts)
    return pal


def compute_dE00(img_name):
    orig = np.array(Image.open(os.path.join(RESULT_DIR, 'outputs', 'original', f'{img_name}.jpg')))
    lab_ref = rgb2lab(orig.astype(np.float64) / 255.0)
    d = {}
    for m in [x[0] for x in TOP + BOT if x[0] != 'original']:
        out = np.array(Image.open(os.path.join(RESULT_DIR, 'outputs', m, f'{img_name}.jpg')))
        d[m] = float(deltaE_ciede2000(lab_ref, rgb2lab(out.astype(np.float64) / 255.0)).mean())
    return d


def draw_palette_grid(ax, colors, counts):
    """Square palette grid with per-block % labels (bottom-right, white bg)."""
    n = len(colors)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    fracs = counts / counts.sum()
    ax.set_xlim(0, cols); ax.set_ylim(0, rows)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(False)
    for i in range(n):
        r = rows - 1 - i // cols
        c = i % cols
        face = np.array(colors[i]) / 255.0
        ax.add_patch(FancyBboxPatch((c + 0.04, r + 0.04), 0.92, 0.92,
                                    boxstyle='round,pad=0.02,rounding_size=0.10',
                                    facecolor=face, edgecolor='white', linewidth=1.0))
        if fracs[i] >= 0.01:
            ax.text(c + 0.88, r + 0.07, f'{fracs[i]*100:.0f}%',
                    fontsize=4.0, ha='right', va='bottom', color='black',
                    bbox=dict(boxstyle='round,pad=0.06', facecolor='white',
                              edgecolor='none', alpha=0.7))
    for i in range(n, rows * cols):
        r = rows - 1 - i // cols
        c = i % cols
        ax.add_patch(FancyBboxPatch((c + 0.04, r + 0.04), 0.92, 0.92,
                                    boxstyle='round,pad=0.02,rounding_size=0.10',
                                    facecolor='#F0F0F0', edgecolor='white', linewidth=1.0))


def make_mosaic(img_name, palettes, out_dir, dpi=160):
    dE = compute_dE00(img_name)
    img_id = img_name.replace('kodim', '').lstrip('0').zfill(2) or '01'

    fig = plt.figure(figsize=(12.5, 8.8))
    # 顶部简洁标题
    fig.suptitle(f'#{img_id}  {img_name}    K={K}', fontsize=15, y=0.972)

    gs = GridSpec(4, 4, hspace=0.38, wspace=0.16,
                  height_ratios=[3.4, 0.85, 3.4, 0.85],
                  left=0.04, right=0.96, top=0.94, bottom=0.03)

    def cell_title(method, is_ours):
        """Title text: label + dE00 (right side). SOM gets (63 colors)."""
        if method == 'original':
            return 'Original', False
        txt = f'{LABEL[method]}   $\\Delta E_{{00}}$={dE[method]:.3f}'
        if method == 'som':
            txt = f'{LABEL[method]} (63 colors)   $\\Delta E_{{00}}$={dE[method]:.3f}'
        return txt, is_ours

    def draw_cell(img_row, bar_row, col, method, is_ours, show_title=True):
        ax = fig.add_subplot(gs[img_row, col])
        out = Image.open(os.path.join(RESULT_DIR, 'outputs', method, f'{img_name}.jpg'))
        ax.imshow(np.array(out))
        ax.set_xticks([]); ax.set_yticks([])
        ec = GREEN if is_ours else '#999999'
        lw = 2.5 if is_ours else 0.8
        for spine in ax.spines.values():
            spine.set_edgecolor(ec); spine.set_linewidth(lw)
        if show_title:
            txt, is_ours = cell_title(method, is_ours)
            tcolor = GREEN if is_ours else 'black'
            ax.set_title(txt, fontsize=10.5, pad=6, color=tcolor)
        colors, counts = palettes[img_name][method]
        axb = fig.add_subplot(gs[bar_row, col])
        draw_palette_grid(axb, colors, counts)

    for col, (m, ours) in enumerate(TOP):
        draw_cell(0, 1, col, m, ours)
    for col, (m, ours) in enumerate(BOT):
        draw_cell(2, 3, col, m, ours, show_title=(col != 0))

    out_path = os.path.join(out_dir, f'{img_name}_mosaic.png')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    return out_path


def main():
    palettes = load_palettes()
    mosaic_dir = os.path.join(RESULT_DIR, 'mosaics')
    os.makedirs(mosaic_dir, exist_ok=True)
    path = make_mosaic('kodim01', palettes, mosaic_dir)
    print('MOSAIC:', path)


if __name__ == '__main__':
    main()