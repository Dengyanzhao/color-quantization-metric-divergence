# -*- coding: utf-8 -*-
"""交叉校验：用官方 CQ100 的 MSE 金标准表验证我们的处理流程。

做法：
  用 PIL 内置的 MEDIANCUT / FASTOCTREE 在官方 PPM 上量化，
  计算 RGB 空间的逐图 MSE，与官方 MSE_RGB_{K}.xlsx 的 MC / OCT 列对比。

注意：PIL 的实现细节（切分规则、代表色取整）与原文所用实现未必逐位一致，
因此这里看的是「量级是否吻合、逐图趋势是否一致」，而不是要求数值完全相等。
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
import numpy as np
import pandas as pd
from PIL import Image

PPM_DIR = f'{_DATASETS}/cq100_official/CQ100'
XLSX_DIR = f'{_DATASETS}/cq100_official'
OUT = f'{_ROOT}/data/crosscheck_mc_oct.csv'


def load_u8(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.float64)


def mse(a, b):
    return float(((a - b) ** 2).mean())


def quant_pil(img, k, method):
    return np.asarray(img.quantize(colors=k, method=method,
                                   dither=Image.Dither.NONE).convert('RGB'), dtype=np.float64)


def main():
    ks = [4, 16, 64, 256]
    # 官方列名 -> (PIL method, 我们算的 MSE)
    targets = {'MC': Image.Quantize.MEDIANCUT, 'OCT': Image.Quantize.FASTOCTREE}

    rows = []
    rng = np.random.default_rng(0)
    for k in ks:
        gold = pd.read_excel(os.path.join(XLSX_DIR, 'MSE_RGB_%d.xlsx' % k))
        gold = gold.set_index('Image')
        names = list(gold.index)
        for name in names:
            p = os.path.join(PPM_DIR, name + '.ppm')
            if not os.path.exists(p):
                print('missing', name, flush=True)
                continue
            with Image.open(p) as im0:
                img = im0.convert('RGB')
                ref = np.asarray(img, dtype=np.float64)
                for col, meth in targets.items():
                    q = quant_pil(img, k, meth)
                    rows.append((name, k, col, mse(ref, q), float(gold.loc[name, col])))
        sub = [r for r in rows if r[1] == k]
        print('K=%-4d done, %d rows' % (k, len(sub)), flush=True)

    df = pd.DataFrame(rows, columns=['image', 'k', 'method', 'mse_ours', 'mse_gold'])
    df['ratio'] = df['mse_ours'] / df['mse_gold']
    df['log_ratio'] = np.log10(df['ratio'])
    df.to_csv(OUT, index=False, encoding='utf-8')

    print('\n=== 我们的 MSE / 官方 MSE 比值 ===')
    for (k, m), g in df.groupby(['k', 'method']):
        r = g['ratio']
        print('K=%-4d %-4s  n=%3d  median_ratio=%.3f  IQR=[%.3f, %.3f]  corr=%.4f'
              % (k, m, len(g), r.median(), r.quantile(.25), r.quantile(.75),
                 np.corrcoef(g['mse_ours'], g['mse_gold'])[0, 1]))
    print('\nwrote', OUT)


if __name__ == '__main__':
    sys.exit(main())
