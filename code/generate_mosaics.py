# -*- coding: utf-8 -*-
"""批量生成 24 张 Kodak 拼接图（Mosaic）。
从 outputs/ 读量化图像 + palettes.csv 读调色板。
布局: 4 列 × 2 行，标题行显示图名+K+CE dE00。
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

import numpy as np, cv2, os, math, csv, sys

ROOT = f'{_ROOT}/results'
OUTPUTS = os.path.join(ROOT, 'outputs')
MOSAICS = os.path.join(ROOT, 'mosaics')
PAL_CSV = os.path.join(ROOT, 'palettes.csv')
W, H = 320, 240
LABEL_H = 34; PAL_CELL_H = 170; TITLE_H = 56; GAP = 8; PAL_MARGIN = 8; PAL_GAP = 3
K = 64

# 方法顺序（4 列 × 2 行）
ROW1 = ['original', 'ce_kmeans', 'mediancut', 'fastoctree']
ROW2 = ['original', 'kmeans', 'maxcoverage', 'som']
ALL_METHODS = list(dict.fromkeys(ROW1 + ROW2))  # 去重保序

# 方法显示名 + 边框色
META = {
    'original': ('Original (24-bit)', (180, 180, 180)),
    'ce_kmeans': ('CE-KMeans (Ours)', (0, 255, 128)),
    'mediancut': ('Median Cut', (120, 120, 120)),
    'fastoctree': ('Fast Octree', (120, 120, 120)),
    'kmeans': ('K-means (RGB)', (120, 120, 120)),
    'maxcoverage': ('Max Coverage', (120, 120, 120)),
    'som': ('SOM', (120, 120, 120)),
}


def rd(fn):
    return cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)


def overlay_label(img, text, color, border):
    out = img.copy()
    bar = np.zeros((LABEL_H, img.shape[1], 3), dtype=np.uint8); bar[:] = (25, 25, 25)
    out[:LABEL_H] = (out[:LABEL_H] * (1 - 0.72) + bar * 0.72).astype(np.uint8)
    cv2.putText(out, text, (10, LABEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), border, 3)
    return out


def draw_palette_cell(palette, pcts, width=W, height=PAL_CELL_H):
    cell = np.zeros((height, width, 3), dtype=np.uint8); cell[:] = (18, 18, 18)
    K = len(palette)
    if K == 0:
        cv2.rectangle(cell, (0, 0), (width - 1, height - 1), (255, 255, 255), 2)
        return cell
    aw = width - 2 * PAL_MARGIN; ah = height - 2 * PAL_MARGIN
    cols = max(1, int(round(math.sqrt(K * aw / ah))))
    rows = math.ceil(K / cols)
    cw = (aw - (cols - 1) * PAL_GAP) / cols
    ch = (ah - (rows - 1) * PAL_GAP) / rows
    for i in range(K):
        c = palette[i]; p = pcts[i]
        r, cc = divmod(i, cols)
        x0 = int(PAL_MARGIN + cc * (cw + PAL_GAP)); y0 = int(PAL_MARGIN + r * (ch + PAL_GAP))
        x1 = int(x0 + cw); y1 = int(y0 + ch)
        cell[y0:y1, x0:x1] = tuple(reversed(c.tolist()))
        cv2.rectangle(cell, (x0, y0), (x1 - 1, y1 - 1), (60, 60, 60), 1)
        text = f'{p:.1f}%'
        lum = 0.299 * float(c[2]) + 0.587 * float(c[1]) + 0.114 * float(c[0])
        tcolor = (0, 0, 0) if lum > 150 else (255, 255, 255)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        if cw >= tw + 2 and ch >= th + 2:
            tx = x0 + (cw - tw) / 2; ty = y0 + (ch + th) / 2
            cv2.putText(cell, text, (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, tcolor, 1, cv2.LINE_AA)
    cv2.rectangle(cell, (0, 0), (width - 1, height - 1), (255, 255, 255), 3)
    return cell


def draw_reference_area(height=PAL_CELL_H):
    area = np.zeros((height, W, 3), dtype=np.uint8); area[:] = (18, 18, 18)
    text = 'Original (reference)'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(area, text, ((W - tw) // 2, (height + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(area, (0, 0), (W - 1, height - 1), (255, 255, 255), 3)
    return area


def main():
    csv.field_size_limit(10000000)
    with open(PAL_CSV, encoding='utf-8') as f:
        pal_rows = {(r[0], r[1]): (eval(r[3]), eval(r[4])) for r in csv.reader(f) if r[0] != 'image'}

    card_h = H + PAL_CELL_H
    total_w = 4 * W
    # 按文件名排序（kodim01~24）
    imgs = sorted([f for f in os.listdir(os.path.join(OUTPUTS, 'original')) if f.endswith('.jpg')])
    for fi, fn in enumerate(imgs):
        img_name = fn.replace('.jpg', '')
        # 读取所有方法的图像
        method_cards = {}
        for m in ALL_METHODS:
            img = rd(os.path.join(OUTPUTS, m, fn))
            if img is None:
                img = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            label, border = META[m]
            tagged = overlay_label(img, label, (255, 255, 255), border)
            # 调色板色块
            if (img_name, m) in pal_rows:
                rgb_list, cnt_list = pal_rows[(img_name, m)]
                pal = np.array(rgb_list).reshape(-1, 3).astype(np.uint8)
                cnts = np.array(cnt_list, dtype=np.float64)
                pcts = cnts / cnts.sum() * 100
                pal_cell = draw_palette_cell(pal, pcts)
            else:
                pal_cell = draw_reference_area()
            card = np.zeros((card_h, W, 3), dtype=np.uint8)
            card[:H] = tagged; card[H:] = pal_cell
            method_cards[m] = card

        # 拼接
        canvas_h = TITLE_H + 2 * card_h + GAP
        canvas = np.zeros((canvas_h, total_w, 3), dtype=np.uint8); canvas[:] = (8, 8, 8)
        title = f'{img_name}    K={K}    CE-KMeans method'
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(canvas, title, ((total_w - tw) // 2, TITLE_H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        for idx, m in enumerate(ROW1 + ROW2):
            r, c = divmod(idx, 4)
            y0 = TITLE_H + r * (card_h + GAP)
            canvas[y0:y0 + card_h, c * W:(c + 1) * W] = method_cards[m]

        out_fn = os.path.join(MOSAICS, f'{img_name}_mosaic.png')
        cv2.imencode('.png', canvas)[1].tofile(out_fn)
        if (fi + 1) % 6 == 0 or fi == len(imgs) - 1:
            print(f'[{fi+1}/{len(imgs)}] {img_name}')
    print(f'Done. 共 {len(imgs)} 张拼接图')


if __name__ == '__main__':
    main()