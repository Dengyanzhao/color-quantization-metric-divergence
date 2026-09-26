# -*- coding: utf-8 -*-
"""生成正式主观实验刺激材料（2AFC, 强制二选一）。

设计（见 PROTOCOL.md）：
  因素 A 方法对  : CE-KMeans vs K-means
  因素 B 图像内容: 6 类（人像/建筑/天空/风景/纹理/夜景）
  因素 C 观看条件: near（原分辨率）/ far（4x 下采样模拟观看距离）
  主试次 = 3 K 值 x 6 图 x 2 条件 = 36
  控制试次 = 8（CE vs CE，同图相同）
  合计 44 试次/被试

刺激源（两种模式，自动选择）：
  MODE='reuse' : 复用 results/outputs/ 里已渲染的 K=64 结果（零计算，但只有 K=64）
  MODE='render': 直接调用量化器在 384x256 上重新量化，得到 K=16/64/256 全套
                 （约 5 分钟，推荐）

与旧版 generate_pairs.py 的区别：
  1) 无"S 差不多"选项 —— 强制二选一
  2) 按 K 分层（16/64/256），旧版只有 K=64
  3) 加入观看距离条件
  4) 控制试次增至 8 个，用于检测 side bias
  5) 输出 trials.csv（盲）+ key.csv（揭盲）

用法: python generate_pairs.py [--mode render|reuse]
输出: pairs/   trials.csv   key.csv   answer_sheet.csv
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
import os
import sys
import csv
import random
import argparse

from PIL import Image
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'paper', 'code'))

# ---------------- 配置 ----------------
ROOT = f'{_ROOT}'
OUTS = os.path.join(ROOT, 'results', 'outputs')
KODAK = os.path.join(ROOT, 'data', 'kodak')
OUTDIR = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(OUTDIR, 'pairs')

K_VALUES = [16, 64, 256]
CONTENT = {
    'portrait':  'kodim01',
    'building':  'kodim05',
    'sky':       'kodim06',
    'landscape': 'kodim16',
    'texture':   'kodim23',
    'night':     'kodim24',
}
N_CONTROL = 8
SEED = 20260923
W, H = 384, 256
GAP = 16

random.seed(SEED)


# ---------------- 量化渲染 ----------------
def _find_kodak(img_name):
    for f in os.listdir(KODAK):
        if f.lower().startswith(img_name):
            return os.path.join(KODAK, f)
    raise FileNotFoundError(img_name)


def render_all():
    """在实验分辨率下渲染 CE-KMeans 与 K-means 的全部 K 值。

    返回 cache[(img_name, method, k)] = BGR uint8 数组
    """
    from PIL import Image as PILImage
    from sklearn.cluster import KMeans

    import ce_kmeans as ce

    cache = {}
    for cname, img in CONTENT.items():
        path = _find_kodak(img)
        src = PILImage.open(path).convert('RGB').resize((W, H), PILImage.LANCZOS)
        arr = np.asarray(src)

        for k in K_VALUES:
            # CE-KMeans
            t = ce.ce_kmeans(src, k)
            cache[(img, 'ce_kmeans', k)] = cv2.cvtColor(np.asarray(t), cv2.COLOR_RGB2BGR)

            # K-means（RGB 空间，与论文 baseline 一致；k-means++ 三次重启取最优）
            X = arr.reshape(-1, 3).astype(np.float64)
            best, best_inertia = None, np.inf
            for seed in (0, 1, 2):
                km = KMeans(n_clusters=k, init='k-means++', n_init=1,
                            random_state=seed, max_iter=300, tol=1e-4)
                lab = km.fit_predict(X)
                if km.inertia_ < best_inertia:
                    best_inertia, best = km.inertia_, lab
            out = km.cluster_centers_[best].clip(0, 255).astype(np.uint8)
            out = out.reshape(H, W, 3)
            cache[(img, 'kmeans', k)] = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        print(f'  rendered {img} ({cname})')
    return cache


def reuse_all():
    """复用 results/outputs/ 的已渲染结果（仅 K=64）。"""
    cache = {}
    for cname, img in CONTENT.items():
        for method in ('ce_kmeans', 'kmeans'):
            p = os.path.join(OUTS, method, f'{img}.jpg')
            if not os.path.exists(p):
                p = os.path.join(OUTS, method, f'{img}.png')
            im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            cache[(img, method, 64)] = im
    return cache


# ---------------- 合成 ----------------
def apply_viewing(im, downsample=4):
    """观看距离模拟：4x 双线性下采样后回放（对应论文 viewing-distance 协议）。"""
    h, w = im.shape[:2]
    small = cv2.resize(im, (w // downsample, h // downsample),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def make_side_by_side(left, right):
    h, w = left.shape[:2]
    canvas = np.full((h, 2 * w + GAP, 3), 128, dtype=np.uint8)
    canvas[:, :w] = left
    canvas[:, w + GAP:] = right
    for label, x0 in (('A', 8), ('B', w + GAP + 8)):
        cv2.rectangle(canvas, (x0 - 4, 4), (x0 + 24, 34), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x0, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['render', 'reuse'], default='render')
    args = ap.parse_args()

    os.makedirs(PAIRS, exist_ok=True)

    print(f'刺激源模式: {args.mode}')
    if args.mode == 'render':
        cache = render_all()
        k_values = K_VALUES
    else:
        cache = reuse_all()
        k_values = [64]

    trials, key = [], []
    tid = 0

    def add(img_name, k, viewing, m_left, m_right):
        nonlocal tid
        tid += 1
        left = cache[(img_name, m_left, k)]
        right = cache[(img_name, m_right, k)]
        if viewing == 'far':
            left, right = apply_viewing(left), apply_viewing(right)
        canvas = make_side_by_side(left, right)
        fn = f'trial_{tid:03d}.png'
        cv2.imencode('.png', canvas)[1].tofile(os.path.join(PAIRS, fn))
        is_ctrl = int(m_left == m_right)
        trials.append({'trial_id': tid, 'file': fn, 'k': k,
                       'viewing': viewing, 'is_control': is_ctrl})
        key.append({'trial_id': tid, 'file': fn, 'image': img_name, 'k': k,
                    'viewing': viewing, 'left': m_left, 'right': m_right,
                    'is_control': is_ctrl})

    # 主试次
    for k in k_values:
        for cname, img in CONTENT.items():
            for viewing in ('near', 'far'):
                if random.random() < 0.5:
                    add(img, k, viewing, 'ce_kmeans', 'kmeans')
                else:
                    add(img, k, viewing, 'kmeans', 'ce_kmeans')

    # 控制试次
    ctrl_imgs = list(CONTENT.values())
    for i in range(N_CONTROL):
        img = ctrl_imgs[i % len(ctrl_imgs)]
        k = k_values[i % len(k_values)]
        add(img, k, 'near', 'ce_kmeans', 'ce_kmeans')

    # 打乱顺序 + 重编号（保证 trial_id 与呈现顺序一致）
    # 注意：不能直接原地 os.replace —— 会产生覆盖冲突（如 5->1 覆盖尚未处理的 1）
    order = list(range(len(trials)))
    random.shuffle(order)
    trials = [trials[i] for i in order]
    key = [key[i] for i in order]

    staging = os.path.join(OUTDIR, '_staging')
    os.makedirs(staging, exist_ok=True)
    # 第一步：全部移到 staging，按新编号命名
    for new_id, t in enumerate(trials, start=1):
        src = os.path.join(PAIRS, f"trial_{t['trial_id']:03d}.png")
        dst = os.path.join(staging, f'trial_{new_id:03d}.png')
        os.replace(src, dst)
    # 第二步：搬回 pairs/
    for new_id in range(1, len(trials) + 1):
        os.replace(os.path.join(staging, f'trial_{new_id:03d}.png'),
                   os.path.join(PAIRS, f'trial_{new_id:03d}.png'))
    os.rmdir(staging)

    for new_id, (t, kk) in enumerate(zip(trials, key), start=1):
        t['trial_id'] = kk['trial_id'] = new_id
        t['file'] = kk['file'] = f'trial_{new_id:03d}.png'

    # 完整性校验：确认生成的文件数与试次数一致
    got = sorted(os.listdir(PAIRS))
    expect = sorted(f'trial_{i:03d}.png' for i in range(1, len(trials) + 1))
    if got != expect:
        missing = set(expect) - set(got)
        raise RuntimeError(f'刺激图缺失 {len(missing)} 张: '
                           f'{sorted(missing)[:5]}...')

    fields_t = ['trial_id', 'file', 'k', 'viewing', 'is_control']
    fields_k = ['trial_id', 'file', 'image', 'k', 'viewing', 'left', 'right',
                'is_control']
    with open(os.path.join(OUTDIR, 'trials.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields_t)
        w.writeheader()
        w.writerows(trials)
    with open(os.path.join(OUTDIR, 'key.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields_k)
        w.writeheader()
        w.writerows(key)

    with open(os.path.join(OUTDIR, 'answer_sheet.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['observer', 'trial_id', 'choice'])
        for t in trials:
            w.writerow(['', t['trial_id'], ''])

    n_main = sum(1 for t in trials if not t['is_control'])
    print(f'生成 {len(trials)} 试次 -> {PAIRS}/')
    print(f'  主试次 {n_main} + 控制试次 {N_CONTROL}')
    print(f'  K 分层: {k_values}')
    print('  trials.csv（盲）+ key.csv（揭盲，勿给被试）+ answer_sheet.csv')


if __name__ == '__main__':
    main()
