# -*- coding: utf-8 -*-
"""生成论文风格拼接图（cv2 绘制，参考 resort_by_advantage.py 风格）。
布局: 4 列 × 2 行，每格 = 带标签图像 + 调色板色块网格。
方法: Original / CE-KMeans(Ours) / MedianCut / FastOctree / K-means / SOM
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
import cv2, os, math, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ce_kmeans import ce_kmeans
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn.cluster import KMeans

ROOT = f'{_DATASETS}/kodak'
OUT = f'{_ROOT}/figures/fig12_methods_comparison.png'
K = 64
W, H = 320, 240
LABEL_H = 34; PAL_CELL_H = 170; TITLE_H = 56; GAP = 8; PAL_MARGIN = 8; PAL_GAP = 3


def rd(fn):
    return cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)


def overlay_label(img, text, color, border):
    out = img.copy()
    bar = np.zeros((LABEL_H, img.shape[1], 3), dtype=np.uint8); bar[:] = (25, 25, 25)
    alpha = 0.72
    out[:LABEL_H] = (out[:LABEL_H] * (1 - alpha) + bar * alpha).astype(np.uint8)
    cv2.putText(out, text, (10, LABEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), border, 3)
    return out


def quantize(img_pil, method):
    if method == 'ce':
        return ce_kmeans(img_pil, K)
    elif method == 'mc':
        return img_pil.quantize(colors=K, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert('RGB')
    elif method == 'fast':
        return img_pil.quantize(colors=K, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).convert('RGB')
    elif method == 'km':
        arr = np.array(img_pil).reshape(-1, 3).astype(np.float32)
        rng = np.random.default_rng(42)
        km = KMeans(n_clusters=K, n_init=3, random_state=42, max_iter=100)
        km.fit(arr[rng.choice(len(arr), min(50000, len(arr)), replace=False)])
        return Image.fromarray(km.cluster_centers_.round().astype(np.uint8)[km.predict(arr)].reshape(img_pil.size[1], img_pil.size[0], 3))
    elif method == 'som':
        arr = np.array(img_pil).reshape(-1, 3).astype(np.float64)
        rng = np.random.default_rng(42)
        W_init = rng.uniform(0, 255, (K, 3))
        pixels = arr[rng.choice(len(arr), min(30000, len(arr)), replace=False)]
        sigma = max(1.0, K / 8.0)
        for t in range(200):
            alpha = 0.3 * (1.0 - t / 200)
            idx = rng.integers(0, len(pixels))
            x = pixels[idx]
            d = np.linalg.norm(W_init - x, axis=1)
            bmu = int(np.argmin(d))
            dists = np.minimum(np.abs(np.arange(K) - bmu), K - np.abs(np.arange(K) - bmu))
            h = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))
            W_init += alpha * h[:, None] * (x - W_init)
        Wc = W_init.round().astype(np.uint8)
        labels = np.argmin(np.linalg.norm(arr[:, None, :] - W_init[None, :, :], axis=2), axis=1)
        return Image.fromarray(Wc[labels].reshape(img_pil.size[1], img_pil.size[0], 3))
    return None


def extract_palette(img_pil, k):
    q = img_pil.convert('P', palette=Image.Palette.ADAPTIVE, colors=k)
    pal = np.array(q.getpalette()[:k * 3]).reshape(-1, 3).astype(np.uint8)
    idx = np.array(q).ravel()
    max_idx = int(idx.max())
    pal = pal[:max_idx + 1]
    counts = np.bincount(idx, minlength=max_idx + 1).astype(np.float64)
    order = np.argsort(-counts)
    return pal[order], (counts[order] / counts.sum() * 100)


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
    fn = os.path.join(ROOT, 'kodim01.png')
    orig = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)
    orig = cv2.resize(orig, (W, H), interpolation=cv2.INTER_AREA)
    orig_pil = Image.open(fn).convert('RGB')

    # 量化所有方法
    methods = [
        ('CE-KMeans (Ours)', 'ce', (0, 255, 128)),
        ('Median Cut', 'mc', (120, 120, 120)),
        ('Fast Octree', 'fast', (120, 120, 120)),
        ('K-means', 'km', (120, 120, 120)),
        ('SOM', 'som', (120, 120, 120)),
    ]
    results = []
    for name, key, border in methods:
        sys.stdout.write(f'Quantizing {name}... ')
        sys.stdout.flush()
        out_pil = quantize(orig_pil, key)
        out_arr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
        out_img = cv2.resize(out_arr, (W, H), interpolation=cv2.INTER_AREA)
        out_img = overlay_label(out_img, name, (255, 255, 255), border)
        pal, pcts = extract_palette(out_pil, K)
        pal_cell = draw_palette_cell(pal, pcts)
        card = np.zeros((H + PAL_CELL_H, W, 3), dtype=np.uint8)
        card[:H] = out_img; card[H:] = pal_cell
        results.append(card)
        sys.stdout.write('done\n')

    # 构建拼接图
    total_w = 4 * W
    card_h = H + PAL_CELL_H
    orig_label = overlay_label(orig.copy(), 'Original (24-bit)', (255, 255, 255), (180, 180, 180))
    orig_card = np.zeros((card_h, W, 3), dtype=np.uint8)
    orig_card[:H] = orig_label; orig_card[H:] = draw_reference_area()

    # 第一行: Original | CE-KMeans | Median Cut | Fast Octree
    row1 = [orig_card, results[0], results[1], results[2]]
    # 第二行: Original | K-means | SOM | 指标说明
    info_card = np.zeros((card_h, W, 3), dtype=np.uint8)
    info_card[:] = (18, 18, 18)
    # 计算各方法 dE00
    arr_ref = np.array(orig_pil)
    lab_ref = rgb2lab(arr_ref.astype(np.float64) / 255.0)
    lines = []
    for name, key, _ in methods:
        out_pil = quantize(orig_pil, key)
        d = float(deltaE_ciede2000(lab_ref, rgb2lab(np.array(out_pil).astype(np.float64) / 255.0)).mean())
        lines.append(f'{name}: dE00={d:.3f}')
    for li, l in enumerate(lines):
        cv2.putText(info_card, l, (10, 30 + li * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(info_card, (0, 0), (W - 1, card_h - 1), (255, 255, 255), 3)
    row2 = [orig_card, results[3], results[4], info_card]

    # 画布
    canvas_h = TITLE_H + 2 * card_h + GAP
    canvas = np.zeros((canvas_h, total_w, 3), dtype=np.uint8); canvas[:] = (8, 8, 8)
    title = f'kodim01 (portrait, skin tones)    K={K}    CE-KMeans dE00={lines[0].split("=")[1]}'
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(canvas, title, ((total_w - tw) // 2, TITLE_H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    for idx, cell in enumerate(row1 + row2):
        r, c = divmod(idx, 4)
        y0 = TITLE_H + r * (card_h + GAP)
        canvas[y0:y0 + card_h, c * W:(c + 1) * W] = cell

    cv2.imencode('.png', canvas)[1].tofile(OUT)
    print(f'\nSaved: {OUT}')
    print(f'Size: {canvas.shape[1]}x{canvas.shape[0]}')


if __name__ == '__main__':
    main()