# -*- coding: utf-8 -*-
"""压缩论文中的大幅 PNG 插图。

背景：fig11 (4170x3753, 16.7 MB) 与 fig9 (4770x885, 5.0 MB) 为 RGBA 无损 PNG
存储照片类内容，导致 paper.pdf 达 24 MB，可能超出投稿系统的上传限制。
两张图的实际排版宽度不足 7 英寸，故 DPI 远超期刊要求的 300。

处理：
  1) RGBA -> RGB（白底合成，去除 alpha 通道）
  2) 等比缩放到 TARGET_DPI 对应的像素尺寸
  3) 存为 JPEG（quality=95，照片类内容；线稿类应保持 PNG）

注意：LaTeX 编译实际使用的是 paper/latex/figures/（与 paper/figures/ 为两个
独立目录），因此本脚本同时处理两处并保持一致。

输出：paper/latex/figures/*.jpg  与  paper/figures/*.jpg
      （原 PNG 已备份至 paper/archive/figures_original_20260923/）
用法：python compress_figures.py
"""
import os
import sys
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
FIGDIRS = [
    os.path.join(BASE, '..', 'latex', 'figures'),   # LaTeX 实际编译使用
    os.path.join(BASE, '..', 'figures'),            # 整理副本
]

# (文件名, latex 中的排版宽度占比, \linewidth 英寸, 目标 DPI)
# elsarticle preprint 单栏: \textwidth ≈ 6.5 in
LINE_WIDTH_IN = 6.5
TARGET_DPI = 400          # 期刊要求 >= 300，留余量
QUALITY = 95

TARGETS = [
    ('fig11_quantization_results.png', 0.95),
    ('fig9_banding_demo.png', 0.85),
]


def compress(src, frac):
    im = Image.open(src)
    w0, h0 = im.size
    size0 = os.path.getsize(src)

    # RGBA -> RGB（白底）
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    elif im.mode != 'RGB':
        im = im.convert('RGB')

    # 目标像素宽度 = 排版宽度(英寸) * DPI
    target_w = int(round(frac * LINE_WIDTH_IN * TARGET_DPI))
    if target_w < w0:
        target_h = int(round(h0 * target_w / w0))
        im = im.resize((target_w, target_h), Image.LANCZOS)
    else:
        target_w, target_h = w0, h0

    out = os.path.splitext(src)[0] + '.jpg'
    im.save(out, 'JPEG', quality=QUALITY, optimize=True, subsampling=0)
    return (w0, h0, size0), (target_w, target_h, os.path.getsize(out)), out


def main():
    grand0 = grand1 = 0
    for fn, frac in TARGETS:
        for d in FIGDIRS:
            src = os.path.join(d, fn)
            if not os.path.exists(src):
                print(f'[跳过] 不存在: {src}')
                continue
            (w0, h0, s0), (w1, h1, s1), out = compress(src, frac)
            grand0 += s0
            grand1 += s1
            print(f'{os.path.relpath(out, BASE)}')
            print(f'  原: {w0}x{h0}  {s0/1024/1024:.2f} MB  ({w0/(frac*LINE_WIDTH_IN):.0f} DPI)')
            print(f'  新: {w1}x{h1}  {s1/1024/1024:.2f} MB  ({w1/(frac*LINE_WIDTH_IN):.0f} DPI)')
            print(f'  压缩比: {s0/s1:.1f}x\n')
    print(f'合计: {grand0/1024/1024:.2f} MB -> {grand1/1024/1024:.2f} MB '
          f'(省 {100*(1-grand1/grand0):.1f}%)')


if __name__ == '__main__':
    main()

