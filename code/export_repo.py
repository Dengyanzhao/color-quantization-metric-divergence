# -*- coding: utf-8 -*-
"""导出可发布的代码仓库到 paper/repository/。

生成一个自包含、可直接 git init + push 的目录，包含：
  code/              全部实验脚本（含路径脱敏）
  data/              结果数据快照（CSV/TXT，支撑论文每个数字）
  figures/           论文插图
  subjective_test/   主观实验包（脚本与元数据，不含大体积刺激图）
  README.md          仓库说明
  LICENSE            MIT
  requirements.txt   依赖
  .gitignore         忽略规则

不包含：原始数据集（第三方公开数据，体积大）、中间产物、历史备份、
        以及任何包含可识别身份信息的脚本（见 EXCLUDE）。

用法: python export_repo.py
输出: paper/repository/
"""
import ast
import os
import re
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(BASE)
ROOT = os.path.dirname(PAPER)
OUT = os.path.join(PAPER, 'repository')

# ---------------------------------------------------------------------------
# 路径脱敏
#
# 原始脚本硬编码了含作者姓氏的本地绝对路径，直接发布会在双盲评审中泄漏身份，
# 同时使代码对第三方不可运行。这里把这些字面量改写为基于文件位置的相对路径，
# 并按仓库目录结构重映射。
#
# 本文件自身也会被发布，因此匹配用词必须以拆分的代码点形式构造：若在这里写出
# 完整的原始路径字面量，导出的脚本自身就会成为一个泄漏点。
# ---------------------------------------------------------------------------
_A = chr(38988) + chr(33041) + chr(35299) + chr(26500)   # 工作区父目录名
_S = chr(38376)                                           # 作者姓氏

# 匹配用词以 \\uXXXX 转义构造。不能用 chr(代码点)：代码点可能在传输中被改动，
# 已导致正则完全失效（0 命中）。也不得写中文字面量，否则导出的脚本自身泄漏。
_A = '\\u8bba\\u6587\\u89e3\\u6784'.encode('utf-8').decode('unicode_escape')
_S = '\\u9093'.encode('utf-8').decode('unicode_escape')

_ABS_PAT = re.compile(
    "(?<![\\w.])(r?b?)([\"'])(?:D:)([\\\\/]"
    "(?:%s[\\\\/]%s)((?:[\\\\/][^'\"]*)?)?)\\2" % (_A, _S),
    re.S)

# 泄漏检查用词：原始路径中连续出现、可直接标识作者的片段
_LEAK_TOKEN = '%s/%s' % (_A, _S)

# 原目录 -> 仓库内目录（按最长前缀优先匹配）。
# 注意：顶层 /data 是数据集目录，由 _map() 单独处理为仓库外的 datasets/。
_DIR_MAP = [
    ('/paper/latex/figures', '/figures'),
    ('/paper/figures', '/figures'),
    ('/paper/data', '/data'),
    ('/.reasonix/lit', '/data'),
    ('/results', '/results'),
    ('/paper/code', '/code'),
]

# 注入到被改写文件顶部的路径常量定义。
# 必须在模块 docstring 之后插入，否则会切断 docstring、使 __doc__ 变为 None。
_ROOT_DEF = (
    "\n# --- path setup added for public release ---\n"
    "# The original scripts referenced a local absolute path that contained\n"
    "# the author's name; it is now resolved relative to this file instead.\n"
    "# Override with the PROJECT_ROOT env var if needed.\n"
    "#\n"
    "# Raw image datasets are NOT bundled (they are public and large); see the\n"
    "# dataset URLs in README.md. Set DATASETS_DIR to your local copy:\n"
    "#   $env:DATASETS_DIR = 'C:/my/datasets'      # PowerShell\n"
    "#   export DATASETS_DIR=$HOME/datasets        # bash\n"
    "import os as _os\n"
    "\n"
    "_ROOT = _os.environ.get(\n"
    "    'PROJECT_ROOT',\n"
    "    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))\n"
    "\n"
    "_DATASETS = _os.environ.get('DATASETS_DIR', _os.path.join(_ROOT, 'datasets'))\n"
    "\n"
    "# Root of this checkout, where data/ and figures/ live.\n"
    "ROOT = _ROOT\n"
    "\n"
    "# Directory of this script (so sibling modules can be imported by name).\n"
    "_CODE = _os.path.dirname(_os.path.abspath(__file__))\n"
    "# --- end path setup ---\n"
)

# 这些脚本不能发布：它们包含作者可识别的信息（GitHub 用户名、token 文件名等），
# 与双盲要求冲突。功能已由 push_to_github.ps1 覆盖。
EXCLUDE = {'push_repo_via_api.py'}


def _map(rest):
    """把原始根目录之后的剩余路径映射为 (变量名, 后缀)。

    rest 带前导 '/'。返回的后缀也带前导 '/'，并保留原始末尾 '/'
    （许多脚本用 `OUT_FIG + 'fig.pdf'` 拼接，末尾斜杠是语义的一部分）。

      '/data/kodak'          -> ('_DATASETS', '/kodak')
      '/data'                -> ('_DATASETS', '')
      '/data/'               -> ('_DATASETS', '/')
      '/paper/data/x.csv'    -> ('_ROOT', '/data/x.csv')
      '/paper/data/'         -> ('_ROOT', '/data/')
      '/paper/figures/'      -> ('_ROOT', '/figures/')
      '/paper/latex/figures' -> ('_ROOT', '/figures')
      '/.reasonix/lit/a.csv' -> ('_ROOT', '/data/a.csv')
      '/results/outputs'     -> ('_ROOT', '/results/outputs')
      '/results'             -> ('_ROOT', '/results')

    返回 (None, None) 表示无已知映射（原样保留）。
    """
    t = rest.replace('\\', '/')
    trailing = '/' if len(t) > 1 and t.endswith('/') else ''
    core = t[:-1] if trailing else t

    # 数据集：原 /data/<名字>/... -> 仓库外的 datasets/<名字>/...
    if core == '/data':
        return '_DATASETS', trailing
    if core.startswith('/data/'):
        return '_DATASETS', core[5:] + trailing

    # 脚本自身所在目录：sys.path.insert(0, <repo>/code) -> 用 _CODE
    if core == '/paper/code':
        return '_CODE', trailing
    if core.startswith('/paper/code/'):
        return '_CODE', core[len('/paper/code'):] + trailing

    for src, dst in _DIR_MAP:
        s = src.rstrip('/')
        if core == s:
            return '_ROOT', dst + trailing
        if core.startswith(s + '/'):
            # core[len(s):] 本身带前导 '/'，直接拼接即可
            return '_ROOT', dst + core[len(s):] + trailing
    return None, None


def sanitize_py(text):
    """把硬编码的本地绝对路径改写为可移植形式。返回 (新文本, 命中数)。"""
    hits = 0

    def _sub(m):
        nonlocal hits
        hits += 1
        # group(4) = 去掉原始根目录之后的剩余部分（可能为空）
        rest = m.group(4)
        if not rest:                 # 仅匹配到工作区根目录本身
            return "f'{" + "_ROOT" + "}'"
        var, suffix = _map(rest)
        if var is None:
            return m.group(0)        # 无已知映射，原样保留
        # 生成 f'{VAR}<suffix>' —— 闭括号在 VAR 之后、suffix 之前
        return "f'{" + var + "}" + suffix + "'"

    new = _ABS_PAT.sub(_sub, text)
    if hits == 0:
        return text, 0

    lines = new.split('\n')
    n = len(lines)
    # 插入位置：编码声明、模块 docstring 及其空行之后的第一行
    i = 0
    if i < n and lines[i].lstrip().startswith('# -*- coding'):
        i = 1
    while i < n and not lines[i].strip():
        i += 1
    if i < n and lines[i].lstrip()[:3] in ('"""', "'''"):
        q = lines[i].lstrip()[:3]
        if lines[i].count(q) >= 2:
            i += 1                     # 单行 docstring
        else:
            j = i + 1
            while j < n and q not in lines[j]:
                j += 1
            i = (j + 1) if j < n else i
        while i < n and not lines[i].strip():
            i += 1
    lines.insert(i, _ROOT_DEF)
    return '\n'.join(lines), hits


SANITIZED = []


def copy_tree(src, dst, skip_names=(), skip_ext=(), sanitize=False):
    n = 0
    if not os.path.isdir(src):
        return 0
    os.makedirs(dst, exist_ok=True)
    for item in sorted(os.listdir(src)):
        if item in skip_names or item.startswith('.') or item == '__pycache__':
            continue
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            n += copy_tree(s, d, skip_names, skip_ext, sanitize)
        else:
            if os.path.splitext(item)[1].lower() in skip_ext:
                continue
            if sanitize and item.endswith('.py'):
                with open(s, encoding='utf-8') as f:
                    text = f.read()
                text, hits = sanitize_py(text)
                if hits:
                    SANITIZED.append((os.path.relpath(s, ROOT), hits))
                with open(d, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(text)
            else:
                shutil.copy2(s, d)
            n += 1
    return n


README = '''# Metric Divergence in Color Quantization

Code and result data for the paper:

> **Metric Divergence in Color Quantization: Method Ranking and Color Space Selection**

## What this repository contains

| Directory | Contents |
|---|---|
| `code/` | All experiment scripts (quantizers, metric computation, reproduction, analysis) |
| `data/` | Result snapshots (CSV/TXT). Every number in the paper traces to a file here. |
| `figures/` | Paper figures |
| `subjective_test/` | Complete 2AFC subjective-experiment package (protocol, stimulus generator, collection interface, analysis script) |

## Core findings

1. **Color-space choice is metric-dependent.** Across 1,488 quantizations on CQ100 (100 images) and Kodak (24 images), a six-metric panel (VIF, PSNR, SSIM, CIEDE2000, LPIPS-AlexNet, LPIPS-VGG) splits into two camps: five fidelity metrics agree with one another on 75-100% of images, while CIEDE2000 agrees with them on only 13-21%.
2. **The split is predictable.** We formulate the *Alignment Principle*: each metric has a design-matched space `S*(M)`, and a quantizer run in `S*(M)` necessarily wins under `M`. Three falsifiable predictions follow, and all three hold (`code/alignment_principle.py`).
3. **CE-KMeans** embeds the complete CIEDE2000 structured distance into the K-means assignment step, reducing mean CIEDE2000 by 14.9% over RGB K-means.

## Quick start

```bash
pip install -r requirements.txt

# Verify the Alignment Principle (Predictions P1/P2/P3) — reads only data/
python code/alignment_principle.py
```

`alignment_principle.py` reads only the committed `data/` snapshots and needs no
downloads.

## Path setup

Scripts were ported from a local checkout and now resolve all paths relative to
their own location (see the `path setup added for public release` block at the
top of each affected file). Raw image datasets are **not** bundled, so any script
that reads images needs a local copy:

```powershell
$env:DATASETS_DIR = 'C:/my/datasets'      # PowerShell
```
```bash
export DATASETS_DIR=$HOME/datasets        # bash
```

Expected layout under `$DATASETS_DIR`:

| Path | Dataset |
|---|---|
| `kodak/` | Kodak PhotoCD |
| `cq100_official/CQ100/` | CQ100 |
| `bsds300/` | BSDS300 test |
| `div2k/` | DIV2K |

## Key scripts

| Script | Purpose |
|---|---|
| `alignment_principle.py` | Tests the three falsifiable predictions (paper Section 4.3); reads only `data/` |
| `ce_kmeans.py` | CE-KMeans quantizer + training-free iterative palette-size search |
| `repro_maitra_multi.py` | Color-space reproduction over the six-metric panel (needs `$DATASETS_DIR`) |
| `analyze_cross_dataset.py` | CQ100 vs. Kodak consistency |
| `verify_sharma34.py` | Validates the CIEDE2000 implementation against the 34 reference pairs |
| `crosscheck_official_mse.py` | Cross-checks against the CQ100 publisher's own MSE tables |
| `experiment.py` | Baseline comparison (5 algorithms x 5 palette sizes) |
| `bsd100_validate.py`, `div2k_validate.py` | Cross-dataset generalization |

## Data

Results are snapshots produced by the scripts above:

- `maitra_six_metrics.csv` — 1,200 rows: 100 CQ100 images x 4 palette sizes x 3 color spaces x 6 metrics
- `kodak_six_metrics.csv` — 288 rows: the same protocol on Kodak
- `alignment_matrix.csv` — win rates per metric per design-matched space
- `alignment_principle.txt` — full verification output for P1/P2/P3

The original image datasets are **not** included (they are public and large):

- CQ100: Mendeley Data, doi:10.17632/vw5ys9hfxw.3
- Kodak PhotoCD: https://r0k.us/graphics/kodak/
- BSDS300: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/

## Subjective experiment

`subjective_test/` contains a ready-to-run 2AFC protocol (36 main trials + 8 catch trials) with a power analysis, a stimulus generator, a single-file survey page, a LAN collection server, and an analysis script. See `subjective_test/PROTOCOL.md`. The 44 rendered stimuli are not committed (regenerate with `generate_pairs.py --mode render`, about 5 minutes).

## Requirements

Python 3.10+ and the packages in `requirements.txt`. Experiments were run on a single CPU, single-threaded, with fixed random seeds.

## License

MIT (see `LICENSE`). The result data in `data/` may be reused under the same terms.
'''

LICENSE = '''MIT License

Copyright (c) 2026 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

REQUIREMENTS = '''numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-image>=0.21
scikit-learn>=1.3
pillow>=10.0
matplotlib>=3.7
opencv-python>=4.8

# Only needed for the subjective-experiment collection server
flask>=3.0
'''

GITIGNORE = '''__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/

# LaTeX build artifacts
*.aux
*.log
*.out
*.bbl
*.blg
*.fls
*.fdb_latexmk
*.synctex.gz

# Generated by make_survey.py (contains embedded images, ~16 MB)
subjective_test/survey.html

# Rendered stimuli (regenerate with generate_pairs.py --mode render)
subjective_test/pairs/
subjective_test/_staging/

# Collected responses (may contain participant identifiers)
subjective_test/results_submitted/

# Local dataset copies and credentials
datasets/
*_token
.gh_token

# OS
.DS_Store
Thumbs.db
'''


def _purge(path):
    """删除目录树，跳过被锁定的条目（例如 .git/objects 被编辑器监视时）。

    返回未能删除的路径列表。
    """
    skipped = []
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            with os.scandir(path) as it:
                for entry in it:
                    skipped += _purge(entry.path)
            try:
                os.rmdir(path)
            except OSError:
                skipped.append(path)
        else:
            try:
                os.unlink(path)
            except OSError:
                skipped.append(path)
    except OSError:
        skipped.append(path)
    return skipped


def main():
    if os.path.exists(OUT):
        skipped = _purge(OUT)
        if os.path.exists(OUT):
            keep = '.git' if os.path.isdir(os.path.join(OUT, '.git')) else None
            print('!! 无法完全清空 %s' % OUT)
            print('   （可能有进程正锁定该目录，例如编辑器或文件监视器）')
            print('   未删除: %s' % (skipped[:5] if skipped else '无'))
            print('   本次导出会在其上重建工作文件，但请勿直接推送旧的 .git。')
            print()
    os.makedirs(OUT, exist_ok=True)

    total = 0
    total += copy_tree(os.path.join(PAPER, 'code'), os.path.join(OUT, 'code'),
                       skip_names=EXCLUDE, skip_ext={'.pyc'}, sanitize=True)
    total += copy_tree(os.path.join(PAPER, 'data'), os.path.join(OUT, 'data'),
                       skip_names={'backup_old', 'div2k'})
    total += copy_tree(os.path.join(PAPER, 'latex', 'figures'),
                       os.path.join(OUT, 'figures'))
    total += copy_tree(os.path.join(ROOT, 'subjective_test'),
                       os.path.join(OUT, 'subjective_test'),
                       skip_names={'pairs', '_staging', 'results_submitted',
                                   'archive_pre_v2_20260923', 'survey.html',
                                   'analysis.txt', 'analysis_summary.csv'},
                       sanitize=True)

    for name, content in [('README.md', README), ('LICENSE', LICENSE),
                          ('requirements.txt', REQUIREMENTS),
                          ('.gitignore', GITIGNORE)]:
        with open(os.path.join(OUT, name), 'w', encoding='utf-8',
                  newline='\n') as f:
            f.write(content)
        total += 1

    total_bytes = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, _, fs in os.walk(OUT) for f in fs)
    print('导出完成: ' + OUT)
    print('  文件数: ' + str(total))
    print('  总计:   %.2f MB' % (total_bytes / 1048576.0))
    print()
    for sub in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, sub)
        if os.path.isdir(p):
            cnt = sum(len(fs) for _, _, fs in os.walk(p))
            sz = sum(os.path.getsize(os.path.join(dp, f))
                     for dp, _, fs in os.walk(p) for f in fs)
            print('  %-22s %3d files  %6.2f MB' % (sub + '/', cnt, sz / 1048576.0))
        else:
            print('  %-22s      %6.1f KB' % (sub, os.path.getsize(p) / 1024.0))

    print()
    print('脱敏改写: %d 个文件' % len(SANITIZED))
    for rel, hits in SANITIZED:
        print('  %3d  %s' % (hits, rel))

    print()
    print('=== 自检 ===')
    problems = 0

    # 0. 旧 .git 处理：历史提交中可能仍有泄漏内容，必须重新初始化
    old_git = os.path.join(OUT, '.git')
    if os.path.isdir(old_git):
        problems += 1
        print('  [FAIL] 检测到旧的 .git（历史提交中可能仍有路径泄漏）')
        print('         请手动删除该目录后重新运行，再执行 git init。')
    else:
        print('  [ok]   无旧的 .git，可安全 git init')

    # 1. 泄漏检查（排除 .git，其中可能留存历史提交）
    leaks = []
    for dp, dirs, fs in os.walk(OUT):
        if '.git' in dirs:
            dirs.remove('.git')
        for fn in fs:
            fp = os.path.join(dp, fn)
            try:
                with open(fp, encoding='utf-8', errors='replace') as f:
                    if _LEAK_TOKEN in f.read():
                        leaks.append(os.path.relpath(fp, OUT))
            except OSError:
                pass
    if leaks:
        problems += 1
        print('  [FAIL] 仍存在路径泄漏: %s' % leaks[:5])
    else:
        print('  [ok]   无路径泄漏')

    # 2. 语法检查
    n_syntax_fail = 0
    for dp, dirs, fs in os.walk(OUT):
        if '.git' in dirs:
            dirs.remove('.git')
        for fn in sorted(fs):
            if not fn.endswith('.py'):
                continue
            fp = os.path.join(dp, fn)
            try:
                ast.parse(open(fp, encoding='utf-8',
                               errors='replace').read())
            except SyntaxError as e:
                n_syntax_fail += 1
                problems += 1
                print('  [FAIL] 语法错误 %s: %s' % (os.path.relpath(fp, OUT), e))
    if n_syntax_fail == 0:
        print('  [ok]   全部 .py 语法正确')

    # 3. 排除检查
    for name in EXCLUDE:
        if os.path.exists(os.path.join(OUT, 'code', name)):
            problems += 1
            print('  [FAIL] 排除文件仍被导出: code/' + name)
    else:
        print('  [ok]   已排除含身份信息的脚本')

    if problems:
        print()
        print('  !! %d 个问题，请勿推送仓库' % problems)
        sys.exit(1)
    print('  [ok]   全部检查通过')


if __name__ == '__main__':
    main()
