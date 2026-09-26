# -*- coding: utf-8 -*-
"""对齐原理（Alignment Principle）的验证脚本。

为论文 Section 4.3 生成可证伪的三组预测及其检验：
  P1  每个度量 M 在其**设计匹配空间** S*(M) 上的胜出率显著高于其他空间；
  P2  对齐紧密度决定胜出率：与量化目标恒等的度量（PSNR ≡ RGB-MSE 的单调函数）
      给出接近 100% 的胜出率，仅为近似关系的度量给出明显更低的值；
  P3  CIE-XYZ 没有任何指标以它为目标，故其胜出率应远低于随机基线 33.3%。

设计匹配空间 S*(M) 由各度量的**构造定义**给定，不从数据拟合：
  VIF / PSNR / SSIM / LPIPS-Alex / LPIPS-VGG 作用于 RGB 信号 → rgb
  CIEDE2000 是感知色差度量，LUV 的设计目标即感知均匀 → luv

输入 : paper/data/maitra_six_metrics.csv   (CQ100, 100 images, K in {4,16,64,256})
       paper/data/kodak_six_metrics.csv    (Kodak, 24 images,  K in {4,16,64,256})
输出 : paper/data/alignment_principle.txt
       paper/data/alignment_matrix.csv
用法 : python alignment_principle.py
"""
import pandas as pd
import numpy as np
import os
import sys
from io import StringIO

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

SPACES = ['rgb', 'xyz', 'luv']
UP = ['vifp', 'psnr', 'ssim']
DN = ['dE00', 'lpips_alex', 'lpips_vgg']
METRICS = UP + DN

# 设计匹配空间 S*(M)：来自各度量的构造定义
DESIGN_MATCH = {
    'vifp': 'rgb',
    'psnr': 'rgb',
    'ssim': 'rgb',
    'dE00': 'luv',
    'lpips_alex': 'rgb',
    'lpips_vgg': 'rgb',
}

# 对齐紧密度分级：identity = 与量化目标同一量；approx = 近似关系
TIGHTNESS = {
    'psnr': 'identity',
    'ssim': 'approx',
    'dE00': 'approx',
    'vifp': 'approx',
    'lpips_alex': 'approx',
    'lpips_vgg': 'approx',
}

RANDOM_BASELINE = 100.0 / len(SPACES)   # 33.33%


def load(fn):
    df = pd.read_csv(os.path.join(BASE, fn))
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
    return df


def win_matrix(df, k=None):
    """DataFrame[metric, space] = 该空间胜出的图像占比 (%)。

    k=None 时按 K 分层计算再平均，避免"跨 K 取最优"造成的口径虚高，
    使数字与 Table (tab:win) 的逐 K 口径一致。
    """
    if k is None:
        ks = sorted(df.k.unique())
        mats = [win_matrix(df, kk) for kk in ks]
        return sum(mats) / len(mats)

    sub = df[df.k == k]
    imgs = sorted(sub.image.unique())
    M = pd.DataFrame(0.0, index=METRICS, columns=SPACES)
    for img in imgs:
        g = sub[sub.image == img]
        for m in METRICS:
            vals = g.set_index('space')[m].astype(float)
            w = vals.idxmax() if m in UP else vals.idxmin()
            M.loc[m, w] += 1
    return M / len(imgs) * 100


def analyze(label, df, out):
    print('=' * 78, file=out)
    print(f'{label}', file=out)
    print('=' * 78, file=out)

    W = win_matrix(df)
    W['S*(M)'] = [DESIGN_MATCH[m] for m in W.index]
    W['win@S*'] = [W.loc[m, DESIGN_MATCH[m]] for m in W.index]
    W['other_mean'] = [W.loc[m, [s for s in SPACES if s != DESIGN_MATCH[m]]].mean()
                       for m in W.index]
    W['advantage'] = W['win@S*'] - W['other_mean']

    # ---------- P1 ----------
    print('\n[P1] 胜出率矩阵 (% of images, 全 K 汇总)', file=out)
    print(W.round(1).to_string(), file=out)
    ok1 = (W['advantage'] > 0).all()
    print(f'\n  P1 检验: 所有度量在其 S*(M) 上胜出率均高于其他空间 → '
          f'{"成立" if ok1 else "不成立"}', file=out)

    # ---------- P2 ----------
    print('\n[P2] 对齐紧密度 → 胜出率', file=out)
    print(f'{"metric":>11} {"relation":>9} {"S*":>5} {"win@S*":>9}', file=out)
    for m in METRICS:
        print(f'{m:>11} {TIGHTNESS[m]:>9} {DESIGN_MATCH[m]:>5} '
              f'{W.loc[m, "win@S*"]:>8.1f}%', file=out)
    ident = np.mean([W.loc[m, 'win@S*'] for m in METRICS if TIGHTNESS[m] == 'identity'])
    approx = np.mean([W.loc[m, 'win@S*'] for m in METRICS if TIGHTNESS[m] == 'approx'])
    ok2 = ident > approx
    print(f'\n  identity 均值 = {ident:.1f}%   approx 均值 = {approx:.1f}%', file=out)
    print(f'  P2 检验: identity > approx → {"成立" if ok2 else "不成立"}', file=out)

    # ---------- P3 ----------
    print(f'\n[P3] CIE-XYZ（无匹配空间）的胜出率  [随机基线 = {RANDOM_BASELINE:.1f}%]',
          file=out)
    print(f'{"K":>5} ' + ' '.join(f'{m:>11}' for m in METRICS) + f'{"mean":>9}', file=out)
    xyz_means = []
    for k in sorted(df.k.unique()):
        Wk = win_matrix(df, k)
        row = [Wk.loc[m, 'xyz'] for m in METRICS]
        xyz_means.append(np.mean(row))
        print(f'{k:>5} ' + ' '.join(f'{v:>10.1f}%' for v in row)
              + f'{np.mean(row):>8.1f}%', file=out)
    ok3 = all(v < RANDOM_BASELINE for v in xyz_means)
    print(f'  P3 检验: 所有 K 下 XYZ 均低于随机基线 → '
          f'{"成立" if ok3 else "不成立"}', file=out)

    # ---------- K 依赖 ----------
    print('\n[K 依赖] 各度量在其 S*(M) 上的胜出率随 K 变化', file=out)
    print(f'{"K":>5} ' + ' '.join(f'{m:>11}' for m in METRICS) + f'{"mean":>8}', file=out)
    for k in sorted(df.k.unique()):
        Wk = win_matrix(df, k)
        row = [Wk.loc[m, DESIGN_MATCH[m]] for m in METRICS]
        print(f'{k:>5} ' + ' '.join(f'{v:>10.1f}%' for v in row)
              + f'{np.mean(row):>7.1f}%', file=out)
    print(file=out)

    return W


def main():
    cq = load('maitra_six_metrics.csv')
    kd = load('kodak_six_metrics.csv')

    buf = StringIO()
    print('对齐原理（Alignment Principle）验证', file=buf)
    print('=' * 78, file=buf)
    print('S*(M) 由各度量的构造定义给定，不从数据拟合。', file=buf)
    print('三个预测 P1/P2/P3 均为可证伪的定量陈述。\n', file=buf)

    W1 = analyze('CQ100  (100 images, K in {4,16,64,256})', cq, buf)
    print(file=buf)
    W2 = analyze('Kodak  (24 images, K in {4,16,64,256})', kd, buf)

    # 汇总矩阵 CSV
    rows = []
    for ds, W in [('CQ100', W1), ('Kodak', W2)]:
        for m in METRICS:
            rows.append({
                'dataset': ds, 'metric': m, 'S_star': DESIGN_MATCH[m],
                'relation': TIGHTNESS[m],
                'win_rgb': W.loc[m, 'rgb'], 'win_xyz': W.loc[m, 'xyz'],
                'win_luv': W.loc[m, 'luv'], 'win_at_S_star': W.loc[m, 'win@S*'],
            })
    out_csv = pd.DataFrame(rows)
    out_csv.to_csv(os.path.join(BASE, 'alignment_matrix.csv'),
                   index=False, encoding='utf-8-sig')

    text = buf.getvalue()
    with open(os.path.join(BASE, 'alignment_principle.txt'), 'w',
              encoding='utf-8') as f:
        f.write(text)
    print(text)
    print(f'已写出: paper/data/alignment_principle.txt')
    print(f'已写出: paper/data/alignment_matrix.csv')


if __name__ == '__main__':
    main()
