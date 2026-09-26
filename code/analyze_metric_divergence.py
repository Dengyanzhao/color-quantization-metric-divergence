# -*- coding: utf-8 -*-
"""六指标完整分歧矩阵分析。

指标：vifp/psnr/ssim 越大越好；dE00/lpips_alex/lpips_vgg 越小越好。
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

import csv
import collections
import numpy as np

P = f'{_ROOT}/data/maitra_multi_full.csv'
rows = list(csv.DictReader(open(P, encoding='utf-8')))

ks = sorted({int(r['k']) for r in rows})
spaces = ['rgb', 'xyz', 'luv']
imgs = sorted({r['image'] for r in rows})
METRICS = [('vifp', True), ('psnr', True), ('ssim', True),
           ('dE00', False), ('lpips_alex', False), ('lpips_vgg', False)]

print('images=%d ks=%s rows=%d' % (len(imgs), ks, len(rows)))
print('metrics=%s' % [m for m, _ in METRICS])


def pick(im, k, m, higher):
    d = {r['space']: float(r[m]) for r in rows if r['image'] == im and int(r['k']) == k}
    sgn = 1.0 if higher else -1.0
    return max(d, key=lambda sp: sgn * d[sp])


print('\n' + '=' * 96)
print('表 1  各指标下「胜出空间」的逐图计数（100 张）')
print('=' * 96)
hdr = '%-6s' % 'K'
for m, hi in METRICS:
    hdr += '%14s' % ('%s%s' % (m, '↑' if hi else '↓'))
print(hdr)
for k in ks:
    line = '%-6d' % k
    for m, hi in METRICS:
        cnt = collections.Counter(pick(im, k, m, hi) for im in imgs)
        line += '%14s' % ('%d/%d/%d' % (cnt['rgb'], cnt['xyz'], cnt['luv']))
    print(line)
print('(格式 rgb/xyz/luv)')

print('\n' + '=' * 96)
print('表 2  同一张图，六指标是否选出同一空间（不一致率）')
print('=' * 96)
for k in ks:
    dis = 0
    combos = collections.Counter()
    for im in imgs:
        p = {m: pick(im, k, m, hi) for m, hi in METRICS}
        if len(set(p.values())) > 1:
            dis += 1
            combos[tuple(sorted(set(p.values())))] += 1
    print('K=%-4d 不一致 %3d/100 (%.1f%%)' % (k, dis, 100.0 * dis / len(imgs)))
    for c, n in combos.most_common(4):
        print('        %-18s %d 张' % ('/'.join(c), n))

print('\n' + '=' * 96)
print('表 3  指标两两之间的一致性（逐图同一空间的比例，K=64）')
print('=' * 96)
k0 = 64
names = [m for m, _ in METRICS]
print('%-12s' % '' + ''.join('%12s' % n for n in names))
for a, ha in METRICS:
    line = '%-12s' % a
    for b, hb in METRICS:
        agree = sum(1 for im in imgs if pick(im, k0, a, ha) == pick(im, k0, b, hb))
        line += '%12s' % ('%.0f%%' % (100.0 * agree / len(imgs)))
    print(line)

print('\n' + '=' * 96)
print('表 4  各指标均值（K=64），best 已按方向校正')
print('=' * 96)
for m, hi in METRICS:
    means = {sp: np.mean([float(r[m]) for r in rows if int(r['k']) == k0 and r['space'] == sp])
             for sp in spaces}
    best = max(spaces, key=lambda sp: (1 if hi else -1) * means[sp])
    print('%-12s(%s)  ' % (m, '↑' if hi else '↓') +
          '  '.join('%s=%10.4f' % (sp, means[sp]) for sp in spaces) + '   best=%s' % best)
