# -*- coding: utf-8 -*-
"""复现 Maitra 2026 (J. Electronic Imaging 35(2) 023002) 的三色彩空间 k-means 实验。

原文设置：
  - 色彩空间：RGB / CIE-XYZ / CIE-LUV(-HCL)
  - 算法：k-means（k-means++ 初始化），最小化 WCSS
  - 评估：VIF（Sheikh 2006），[0,1]，越大越好
  - 量化级别：K = 4, 16, 64, 256

我们做两处明示的替换（论文中需声明）：
  - 图像集：原文 148 张（附录仅给文字描述，无法精确复原）→ 改用官方 cq100（100 张，可溯源）
  - k-means 实现：原文 Hartigan-Wong/OTQT (R) → scikit-learn（同为 WCSS 目标）

用法：
  python repro_maitra.py --limit 5 --procs 4   # 小规模试跑
  python repro_maitra.py                       # 全量 100 张，16 进程
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

import os
import sys
import csv
import time
import argparse
import collections
import multiprocessing as mp
import numpy as np
from PIL import Image
from skimage import color as skcolor
from sklearn.cluster import KMeans
from sewar.full_ref import vifp as _vifp

CQ100_DIR = f'{_DATASETS}/cq100_official/CQ100'
OUT_DEFAULT = f'{_ROOT}/data/maitra_repro.csv'

SPACES = ['rgb', 'xyz', 'luv']
KS = [4, 16, 64, 256]


def load_rgb01(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.float64) / 255.0


def to_space(rgb, space):
    if space == 'rgb':
        return rgb
    if space == 'xyz':
        return skcolor.rgb2xyz(rgb)
    if space == 'luv':
        return skcolor.rgb2luv(rgb)
    raise ValueError(space)


def from_space(arr, space):
    if space == 'rgb':
        return arr
    if space == 'xyz':
        return skcolor.xyz2rgb(arr)
    if space == 'luv':
        return skcolor.luv2rgb(arr)
    raise ValueError(space)


def quantize(rgb, space, k, seed=0, n_init=1):
    """在指定空间做 k-means 量化，返回 RGB [0,1] 的量化图。

    用「唯一颜色 + 样本权重」等价替代全像素 k-means：
    WCSS = sum_j n_j ||u_j - mu||^2，与对全像素求和完全相同，
    但样本数从 ~393k 降到几万，内存与耗时大幅下降。
    """
    h, w, _ = rgb.shape
    arr = to_space(rgb, space).reshape(-1, 3)
    uniq, inv, counts = np.unique(arr, axis=0, return_inverse=True, return_counts=True)
    km = KMeans(n_clusters=k, init='k-means++', n_init=n_init,
                random_state=seed, max_iter=300, tol=1e-4)
    km.fit(uniq, sample_weight=counts)
    q = km.cluster_centers_[km.labels_[inv]].reshape(h, w, 3)
    return np.clip(from_space(q, space), 0.0, 1.0)


def compute_vif(ref01, test01):
    a = (ref01 * 255.0).round().astype(np.uint8)
    b = (test01 * 255.0).round().astype(np.uint8)
    return float(_vifp(a, b))


def process_image(job):
    """处理一张图的所有 (K, space) 组合。"""
    name, path, ks, spaces, n_init = job
    t0 = time.time()
    rgb = load_rgb01(path)
    out = []
    for k in ks:
        for sp in spaces:
            q = quantize(rgb, sp, k, n_init=n_init)
            out.append((name, k, sp, compute_vif(rgb, q)))
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--ks', default=','.join(str(k) for k in KS))
    ap.add_argument('--spaces', default=','.join(SPACES))
    ap.add_argument('--n-init', type=int, default=1)
    ap.add_argument('--procs', type=int, default=16)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    spaces = [x.strip() for x in args.spaces.split(',') if x.strip()]

    files = sorted(f for f in os.listdir(CQ100_DIR) if f.endswith('.ppm'))
    if args.limit:
        files = files[:args.limit]
    jobs = [(f[:-4], os.path.join(CQ100_DIR, f), ks, spaces, args.n_init) for f in files]
    print('images=%d  ks=%s  spaces=%s  procs=%d  tasks=%d'
          % (len(files), ks, spaces, args.procs, len(jobs)), flush=True)

    rows = []
    t_start = time.time()
    done = 0
    with mp.Pool(processes=args.procs) as pool:
        for res, elapsed in pool.imap_unordered(process_image, jobs):
            rows.extend(res)
            done += 1
            name = res[0][0] if res else '?'
            print('[%3d/%3d] %-28s %.1fs  (elapsed %.1f min)'
                  % (done, len(jobs), name, elapsed, (time.time() - t_start) / 60), flush=True)

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        wtr = csv.writer(fh)
        wtr.writerow(['image', 'k', 'space', 'vif'])
        for r in rows:
            wtr.writerow([r[0], r[1], r[2], '%.6f' % r[3]])
    print('\nwrote %s  (%d rows, %.1f min)' % (args.out, len(rows), (time.time() - t_start) / 60))

    agg = collections.defaultdict(list)
    for name, k, sp, v in rows:
        agg[(k, sp)].append(v)
    print('\n--- mean VIF by (K, space) ---')
    for k in ks:
        print('K=%-4d ' % k + '  '.join('%s=%.4f' % (sp, float(np.mean(agg[(k, sp)])))
                                        for sp in spaces if agg.get((k, sp))))
    print('\n--- winner count per image per K ---')
    for k in ks:
        win = collections.Counter()
        for name in {r[0] for r in rows}:
            vals = {sp: v for n, kk, sp, v in rows if n == name and kk == k}
            if vals:
                win[max(vals, key=vals.get)] += 1
        print('K=%-4d ' % k + '  '.join('%s=%d' % (sp, win.get(sp, 0)) for sp in spaces))
    return 0


if __name__ == '__main__':
    sys.exit(main())
