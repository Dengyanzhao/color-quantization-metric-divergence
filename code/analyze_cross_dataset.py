# -*- coding: utf-8 -*-
"""跨数据集稳健性：把 CQ100 与 Kodak 的六指标分歧结构并列比较。"""

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

FILES = {
    'CQ100': f'{_ROOT}/data/maitra_multi_full.csv',
    'Kodak': f'{_ROOT}/data/kodak_six_metrics.csv',
}
METRICS = [('vifp', True), ('psnr', True), ('ssim', True),
           ('dE00', False), ('lpips_alex', False), ('lpips_vgg', False)]
spaces = ['rgb', 'xyz', 'luv']


def load(p):
    return list(csv.DictReader(open(p, encoding='utf-8')))


def winner(rows, im, k, m, higher):
    d = {r['space']: float(r[m]) for r in rows if r['image'] == im and int(r['k']) == k}
    sgn = 1.0 if higher else -1.0
    return max(d, key=lambda sp: sgn * d[sp])


for name, path in FILES.items():
    rows = load(path)
    ks = sorted({int(r['k']) for r in rows})
    imgs = sorted({r['image'] for r in rows})
    print('=' * 84)
    print('%s   images=%d  rows=%d' % (name, len(imgs), len(rows)))
    print('=' * 84)

    print('%-6s' % 'K' + ''.join('%14s' % m for m, _ in METRICS))
    for k in ks:
        line = '%-6d' % k
        for m, hi in METRICS:
            c = collections.Counter(winner(rows, im, k, m, hi) for im in imgs)
            line += '%14s' % ('%d/%d/%d' % (c['rgb'], c['xyz'], c['luv']))
        print(line)

    print('%-14s' % '不一致率')
    for k in ks:
        dis = sum(1 for im in imgs
                  if len({winner(rows, im, k, m, hi) for m, hi in METRICS}) > 1)
        print('   K=%-4d  %d/%d = %.1f%%' % (k, dis, len(imgs), 100.0 * dis / len(imgs)))

    k0 = ks[-1] if ks else 64
    print('%-14s (K=%d)' % ('dE00 vs 其他', k0))
    for m, hi in METRICS:
        if m == 'dE00':
            continue
        agree = sum(1 for im in imgs if winner(rows, im, k0, 'dE00', False)
                    == winner(rows, im, k0, m, hi))
        print('   dE00 vs %-11s 一致率 %.0f%%' % (m, 100.0 * agree / len(imgs)))
    print()
