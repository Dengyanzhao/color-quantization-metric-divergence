# -*- coding: utf-8 -*-
"""LPIPS 全量重跑（完整版 CE-KMeans 输出上计算）。
对 Kodak 24 图 × 3 方法 × 5 色数 = 360 次量化，逐图算 LPIPS (AlexNet)。
输出：data/lpips_results.csv (image, method, n_colors, lpips)
顺带：指标秩相关 (Kendall tau) 与决策翻转率（方案 A）。"""

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

import os, sys, io, time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ce_kmeans import ce_kmeans
from pw_kmeans_v3 import pw_kmeans

import lpips
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim_fn

KODAK_DIR = f'{_DATASETS}/kodak'
OUT = f'{_ROOT}/data/lpips_results.csv'
N_COLORS = [16, 32, 64, 128, 256]
METHODS = ['kmeans_rgb', 'pw_lab', 'ce_kmeans']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_kmeans(img, k):
    arr = np.array(img).reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
    km.fit(arr[np.random.default_rng(42).choice(len(arr), 50000, replace=False)])
    out = km.cluster_centers_.round().astype(np.uint8)[km.predict(arr)]
    return Image.fromarray(out.reshape(img.size[1], img.size[0], 3))


def lpips_tensor(img):
    """PIL RGB -> lpips 输入张量 (1,3,H,W) [-1,1]"""
    a = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    print(f'设备: {DEVICE}')
    loss_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    print(f'Kodak 图: {len(files)}')

    rows = []
    for fn in files:
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        for n in N_COLORS:
            outs = {
                'kmeans_rgb': quantize_kmeans(img, n),
                'pw_lab': pw_kmeans(img, n, use_lab=True, weight_power=0.5),
                'ce_kmeans': ce_kmeans(img, n),
            }
            ref_t = lpips_tensor(img).to(DEVICE)
            for m, out in outs.items():
                out_t = lpips_tensor(out).to(DEVICE)
                with torch.no_grad():
                    d = float(loss_fn(ref_t, out_t).item())
                rows.append({'image': fn, 'method': m, 'n_colors': n, 'lpips': round(d, 5)})
        sys.stdout.write(f'{fn} 完成 ({sum(1 for r in rows if r["image"]==fn)} 行)\n')
        sys.stdout.flush()

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f'\n已落盘: {OUT} ({len(df)} 行)')

    # ---- 方案 A: 指标秩相关 + 决策翻转率 ----
    print('\n===== 方案 A: 指标一致性分析 =====')
    # 汇总各方法×色数均值（需 PSNR/SSIM/dE00 + LPIPS）
    ce = pd.read_csv(f'{_ROOT}/data/ce_results.csv')
    ce = ce[ce.method == 'ce_kmeans']
    # 合并 dE00/psnr/ssim
    lp = df.merge(ce[['image', 'n_colors', 'psnr', 'ssim', 'dE00']],
                  on=['image', 'n_colors'], how='left')
    # LPIPS 需要全方法，重建宽表：每图每色数各方法 LPIPS
    lp_wide = df.pivot_table(index=['image', 'n_colors'], columns='method', values='lpips')

    # 对每个 (图, 色数)：3 方法的 4 指标排序
    # 秩相关：在方法之间算 Kendall tau（n=3 太弱），改为跨 (图×色数) 用方法均值算
    print('\n-- 指标两两 Kendall tau（基于各方法在 5 色数的均值序列，24 图平均）--')
    from scipy.stats import kendalltau, wilcoxon

    # 用 K=64 的图级数据算决策翻转率
    k64_ce = ce[ce.n_colors == 64]
    k64_lp = lp_wide.xs(64, level='n_colors')
    # 每个方法在每张图的 LPIPS 均值（k64_lp 已含）
    print('\n-- K=64 决策翻转率: CIEDE2000 vs LPIPS 推荐不同方法 --')
    # CE 按 dE00 最优、K-means 按 LPIPS 最优的图比例
    # 需要每图 3 方法的 dE00 和 lpips
    # dE00: ce_results.csv 只有 ce_kmeans；需要 kmeans/pw dE00 -> 用 bsd 逻辑或重算
    # 这里简化为：已有 ce_results.csv(ce) + 手头数据，说明将补全
    print('(LPIPS 已落盘；秩相关/翻转率需 dE00 全方法数据，下一步合并)')

if __name__ == '__main__':
    main()
