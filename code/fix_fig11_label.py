# -*- coding: utf-8 -*-
"""修正 fig11 中的标注错误：kodim05 实为越野摩托车赛，原标注 "(water)" 有误。

坐标来自探测（2470x2223 图）：
  第 4 行标签行主体 y=1783-1801（括号/降部延至 ~1805）
  第 4 行第 1 列标签 x=182-441，中心 x=311

字号通过四行已知宽度反推：DejaVu Sans 19px 时系统性偏小 3%，
故实际约 19.5px。替换词 "racing" 在 19.5px 下宽 256px，与原 252px 几乎等长，
因此覆盖区无需扩大。
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
from PIL import Image, ImageDraw, ImageFont
import matplotlib

sys.stdout.reconfigure(encoding='utf-8')

FONT = os.path.join(os.path.dirname(matplotlib.__file__),
                    'mpl-data', 'fonts', 'ttf', 'DejaVuSans.ttf')

FILES = [
    f'{_ROOT}/figures/fig11_quantization_results.jpg',
    f'{_ROOT}/figures/fig11_quantization_results.jpg',
]

BOX = (150, 1774, 480, 1808)      # 覆盖区，避让上方 y<=1616 与下方图像 y>=1813
CENTER_X = 311
BASELINE_Y = 1800
NEW_TEXT = 'Original (kodim05 (racing))'
SIZE = 19.5

font = ImageFont.truetype(FONT, SIZE)
bb = font.getbbox(NEW_TEXT)
new_w = bb[2] - bb[0]
print(f'字体: DejaVu Sans {SIZE}px')
print(f'新文字: "{NEW_TEXT}"')
print(f'  宽度 {new_w}px (原标注 260px，差 {new_w-260:+d})')
print(f'  居中 x={CENTER_X} -> x 范围 {CENTER_X-new_w//2}-{CENTER_X+new_w//2}')
print(f'  覆盖区 {BOX}  宽 {BOX[2]-BOX[0]}px\n')

assert new_w <= (BOX[2] - BOX[0]), '新文字超出覆盖区'

for p in FILES:
    if not os.path.exists(p):
        print(f'[跳过] {p}')
        continue
    im = Image.open(p).convert('RGB')
    d = ImageDraw.Draw(im)
    d.rectangle(BOX, fill=(255, 255, 255))
    d.text((CENTER_X, BASELINE_Y), NEW_TEXT, font=font, fill=(0, 0, 0),
           anchor='ms')
    im.save(p, 'JPEG', quality=95, optimize=True, subsampling=0)
    print(f'✅ 已修补: {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}'
          f'  ({os.path.getsize(p)/1024/1024:.2f} MB)')
