# -*- coding: utf-8 -*-
"""主观实验数据分析（2AFC）。

输入：
  subjective_test/key.csv          揭盲表（trial -> 左右方法）
  subjective_test/results_submitted/*.txt   各被试提交的答案
                                            （survey_server.py 保存格式）

输出：
  subjective_test/analysis.txt     完整报告
  subjective_test/analysis_summary.csv  主要指标汇总

分析内容：
  1. 数据清理（控制试次 side bias 检测、被试排除）
  2. 主结果：各条件下选择 CE-KMeans 的比例 + 二项检验 vs 0.5
  3. RQ1  人眼整体偏向哪个阵营
  4. RQ2  一致性是否随 K 变化
  5. RQ3  观看条件的影响
  6. 与计算指标的一致性（CIEDE2000 vs LPIPS 各自预测人眼的能力）

用法: python analyze_subjective.py
"""
import os
import sys
import csv
import glob
from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(BASE, 'key.csv')
RESULTS = os.path.join(BASE, 'results_submitted')

# 控制试次排除阈值：若被试在"同图相同"的控制试次上选同一侧 >= 7/8，判为固定作答
CTRL_SIDE_THRESHOLD = 7


def load_key():
    k = pd.read_csv(KEY)
    k.columns = [c.strip().lstrip('\ufeff') for c in k.columns]
    return k


def load_answers():
    """读取 results_submitted/*.txt，返回 DataFrame[observer, trial_id, choice]。

    survey_server.py 的保存格式：
        observer: <name>
        提交时间: <ts>
        编号,选择
        <trial_id>,<choice>
        ...
    """
    rows = []
    files = glob.glob(os.path.join(RESULTS, '*.txt'))
    for fp in files:
        obs = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 2:
                    continue
                try:
                    tid = int(parts[0])
                except ValueError:
                    continue          # 表头或说明行
                rows.append({'observer': obs, 'trial_id': tid,
                             'choice': parts[1].upper()})
    return pd.DataFrame(rows)


def binom_report(k_success, n, label, out):
    """二项检验 vs 0.5，双侧。"""
    if n == 0:
        print(f'  {label}: 无数据', file=out)
        return None
    p_hat = k_success / n
    res = stats.binomtest(int(k_success), int(n), 0.5, alternative='two-sided')
    # 95% Wilson 区间（比正态近似更稳妥）
    z = 1.959964
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    ci = (max(0.0, centre - half), min(1.0, centre + half))
    sig = '*' if res.pvalue < 0.05 else ' '
    print(f'  {label:<28} {p_hat:6.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]  '
          f'n={n:<5} p={res.pvalue:.4f} {sig}', file=out)
    return {'label': label, 'p_ce': p_hat, 'ci_lo': ci[0], 'ci_hi': ci[1],
            'n': n, 'p_value': res.pvalue}


def main():
    key = load_key()
    ans = load_answers()

    out = StringIO()
    print('=' * 78, file=out)
    print('主观实验分析（2AFC）', file=out)
    print('=' * 78, file=out)

    if ans.empty:
        print('\n未找到任何被试数据（results_submitted/ 为空）。', file=out)
        print('请先运行 survey_server.py 采集数据。', file=out)
        text = out.getvalue()
        print(text)
        with open(os.path.join(BASE, 'analysis.txt'), 'w',
                  encoding='utf-8') as f:
            f.write(text)
        return

    # ---------- 合并 ----------
    df = ans.merge(key, on='trial_id', how='inner',
                   suffixes=('', '_key'))
    print(f'\n被试数: {df.observer.nunique()}   总试次数: {len(df)}', file=out)

    # ---------- 控制试次：side bias 检测 ----------
    ctrl = df[df.is_control == 1]
    print(f'\n[数据清理] 控制试次（同图相同）共 {len(ctrl)} 条', file=out)
    bad = []
    for obs, g in ctrl.groupby('observer'):
        a_rate = (g.choice == 'A').mean()
        if max(a_rate, 1 - a_rate) * len(g) >= CTRL_SIDE_THRESHOLD:
            bad.append(obs)
    if bad:
        print(f'  剔除固定作答被试: {bad}', file=out)
    else:
        print('  无被试触发剔除标准', file=out)

    valid = df[~df.observer.isin(bad) & (df.is_control == 0)].copy()

    # ---------- 编码：是否选择了 CE-KMeans ----------
    valid['chose_ce'] = np.where(
        valid.left == 'ce_kmeans', valid.choice == 'A',
        np.where(valid.right == 'ce_kmeans', valid.choice == 'B', np.nan))
    valid = valid.dropna(subset=['chose_ce'])
    valid['chose_ce'] = valid.chose_ce.astype(int)

    summary = []
    print('\n' + '=' * 78, file=out)
    print('主结果：选择 CE-KMeans 的比例（0.5 = 无偏好）', file=out)
    print('=' * 78, file=out)
    print(f'  {"条件":<28} {"   p̂":>6}  {"95% Wilson CI":<18}  n      p值', file=out)

    summary.append(binom_report(valid.chose_ce.sum(), len(valid),
                                'ALL (all conditions)', out))

    print('\n[RQ2] 按 palette size K', file=out)
    for k in sorted(valid.k.unique()):
        sub = valid[valid.k == k]
        summary.append(binom_report(sub.chose_ce.sum(), len(sub),
                                    f'K={k}', out))

    print('\n[RQ3] 按观看条件', file=out)
    for v in sorted(valid.viewing.unique()):
        sub = valid[valid.viewing == v]
        summary.append(binom_report(sub.chose_ce.sum(), len(sub),
                                    f'viewing={v}', out))

    print('\n[交互] K x 观看条件', file=out)
    for (k, v), sub in valid.groupby(['k', 'viewing']):
        summary.append(binom_report(sub.chose_ce.sum(), len(sub),
                                    f'K={k}, viewing={v}', out))

    # ---------- 被试层面一致性 ----------
    print('\n[被试间变异] 每个被试选择 CE-KMeans 的比例', file=out)
    per_obs = valid.groupby('observer').chose_ce.agg(['mean', 'count'])
    for obs, r in per_obs.iterrows():
        print(f'  {obs:<24} {r["mean"]:.3f}  (n={int(r["count"])})', file=out)
    print(f'  被试均值 = {per_obs["mean"].mean():.3f}  '
          f'标准差 = {per_obs["mean"].std():.3f}', file=out)

    # 被试间 t 检验（更保守，把被试当随机效应）
    if len(per_obs) >= 2:
        t, p = stats.ttest_1samp(per_obs['mean'], 0.5)
        print(f'  被试层面单样本 t 检验 (H0: p=0.5): t={t:.3f}, p={p:.4f}',
              file=out)

    # ---------- 落盘 ----------
    text = out.getvalue()
    print(text)
    with open(os.path.join(BASE, 'analysis.txt'), 'w',
              encoding='utf-8') as f:
        f.write(text)
    if summary:
        pd.DataFrame([s for s in summary if s]).to_csv(
            os.path.join(BASE, 'analysis_summary.csv'),
            index=False, encoding='utf-8-sig')
    print(f'已写出: subjective_test/analysis.txt')
    print(f'已写出: subjective_test/analysis_summary.csv')


if __name__ == '__main__':
    main()
