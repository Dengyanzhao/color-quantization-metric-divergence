# -*- coding: utf-8 -*-
"""Maitra 三色彩空间复现 —— 多指标版。

一次量化，同时评估 6 个指标，用于构建「指标分歧矩阵」：
  - vifp        : pixel-domain VIF (sewar, Sheikh 2006 的常用变体)
  - psnr        : 峰值信噪比
  - ssim        : 结构相似度
  - dE00        : 平均 CIEDE2000 色差（越小越好）
  - lpips_alex  : 深度特征感知距离，AlexNet backbone（越小越好）
  - lpips_vgg   : 深度特征感知距离，VGG16 backbone（越小越好）

协议同 repro_maitra.py：官方 cq100，K ∈ {4,16,64,256}，空间 ∈ {rgb,xyz,luv}。

用法：
  python repro_maitra_multi.py --limit 2 --procs 2 --no-lpips
  python repro_maitra_multi.py --procs 6
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
import multiprocessing as mp
import numpy as np
from PIL import Image
from skimage import color as skcolor
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn.cluster import KMeans
from sewar.full_ref import vifp

CQ100_DIR = f'{_DATASETS}/cq100_official/CQ100'
OUT_DEFAULT = f'{_ROOT}/data/maitra_multi.csv'

SPACES = ['rgb', 'xyz', 'luv']
KS = [4, 16, 64, 256]

_LP = {}
_USE_LPIPS = True


def init_worker(use_lpips):
    """每个 worker 只加载一次 LPIPS 模型。"""
    global _USE_LPIPS
    _USE_LPIPS = use_lpips
    if not use_lpips:
        return
    import torch
    torch.set_num_threads(1)
    import lpips
    for net in ('alex', 'vgg'):
        m = lpips.LPIPS(net=net, verbose=False)
        m.eval()
        _LP[net] = m


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
    """唯一颜色 + 样本权重，与全像素 k-means 的 WCSS 等价。"""
    h, w, _ = rgb.shape
    arr = to_space(rgb, space).reshape(-1, 3)
    uniq, inv, counts = np.unique(arr, axis=0, return_inverse=True, return_counts=True)
    km = KMeans(n_clusters=k, init='k-means++', n_init=n_init,
                random_state=seed, max_iter=300, tol=1e-4)
    km.fit(uniq, sample_weight=counts)
    q = km.cluster_centers_[km.labels_[inv]].reshape(h, w, 3)
    return np.clip(from_space(q, space), 0.0, 1.0)


def lpips_pair(net, ref01, test01):
    import torch
    a = torch.from_numpy((ref01 * 2.0 - 1.0).transpose(2, 0, 1)[None].astype(np.float32))
    b = torch.from_numpy((test01 * 2.0 - 1.0).transpose(2, 0, 1)[None].astype(np.float32))
    with torch.no_grad():
        return float(_LP[net](a, b).item())


def evaluate(ref01, test01):
    a = (ref01 * 255.0).round().astype(np.uint8)
    b = (test01 * 255.0).round().astype(np.uint8)
    out = {}
    out['vifp'] = float(vifp(a, b))
    out['psnr'] = float(psnr_fn(a, b, data_range=255))
    out['ssim'] = float(ssim_fn(a, b, channel_axis=2, data_range=255))
    lab_a = rgb2lab(ref01)
    lab_b = rgb2lab(test01)
    out['dE00'] = float(deltaE_ciede2000(lab_a, lab_b).mean())
    if _USE_LPIPS:
        out['lpips_alex'] = lpips_pair('alex', ref01, test01)
        out['lpips_vgg'] = lpips_pair('vgg', ref01, test01)
    return out


def process_image(job):
    name, path, ks, spaces, n_init = job
    t0 = time.time()
    rgb = load_rgb01(path)
    rows = []
    for k in ks:
        for sp in spaces:
            q = quantize(rgb, sp, k, n_init=n_init)
            m = evaluate(rgb, q)
            m['image'], m['k'], m['space'] = name, k, sp
            rows.append(m)
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--ks', default=','.join(str(k) for k in KS))
    ap.add_argument('--spaces', default=','.join(SPACES))
    ap.add_argument('--n-init', type=int, default=1)
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--no-lpips', action='store_true')
    ap.add_argument('--dir', default=CQ100_DIR)
    ap.add_argument('--ext', default='.ppm')
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    spaces = [x.strip() for x in args.spaces.split(',') if x.strip()]
    use_lpips = not args.no_lpips

    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(args.ext))
    if args.limit:
        files = files[:args.limit]
    jobs = [(f[:-len(args.ext)], os.path.join(args.dir, f), ks, spaces, args.n_init)
            for f in files]
    print('images=%d ks=%s spaces=%s procs=%d lpips=%s'
          % (len(files), ks, spaces, args.procs, use_lpips), flush=True)

    cols = ['image', 'k', 'space', 'vifp', 'psnr', 'ssim', 'dE00']
    if use_lpips:
        cols += ['lpips_alex', 'lpips_vgg']

    all_rows = []
    t_start = time.time()
    done = 0
    with mp.Pool(processes=args.procs, initializer=init_worker, initargs=(use_lpips,)) as pool:
        for rows, elapsed in pool.imap_unordered(process_image, jobs):
            all_rows.extend(rows)
            done += 1
            print('[%3d/%3d] %-26s %.1fs  (elapsed %.1f min)'
                  % (done, len(jobs), rows[0]['image'], elapsed, (time.time() - t_start) / 60),
                  flush=True)

    all_rows.sort(key=lambda r: (r['image'], r['k'], r['space']))
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        wtr = csv.writer(fh)
        wtr.writerow(cols)
        for r in all_rows:
            wtr.writerow([r['image'], r['k'], r['space']] +
                         ['%.6f' % r[c] for c in cols[3:]])
    print('\nwrote %s (%d rows, %.1f min)'
          % (args.out, len(all_rows), (time.time() - t_start) / 60), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
