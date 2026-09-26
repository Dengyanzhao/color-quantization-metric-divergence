# -*- coding: utf-8 -*-
"""Deep compression baseline: CompressAI bmshj2018 on Kodak 24 images.
Compare with CE-KMeans (K=64) on PSNR, dE00, and file size.
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

import numpy as np, os, sys, torch, io
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
import compressai
from compressai.zoo import bmshj2018_factorized

sys.path.insert(0, f'{_CODE}')
from ce_kmeans import ce_kmeans, psnr

KODAK_DIR = f'{_DATASETS}/kodak'
device = 'cpu'
K = 64

# 加载模型（quality 1-8，越大质量越高/比特率越高）
models = {}
for q in [2, 4, 6, 8]:
    models[q] = bmshj2018_factorized(quality=q, pretrained=True).eval().to(device)
    print(f'Loaded bmshj2018 quality={q}')


def compress_decompress(model, img_pil):
    """Compress and decompress an image using the model. Returns decoded PIL, byte size."""
    img = np.array(img_pil).astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    h, w = x.shape[2:]
    # pad to multiple of 64
    pad_h, pad_w = (64 - h % 64) % 64, (64 - w % 64) % 64
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    with torch.no_grad():
        out = model(x)
    # crop
    out['x_hat'] = out['x_hat'][:, :, :h, :w]
    # bpp via likelihoods (compressai standard)
    import math
    bpp = sum((torch.log(l).sum() / (-math.log(2))) for l in out['likelihoods'].values())
    bpp = bpp.item() / (1 * h * w)
    # reconstruct
    recon = (out['x_hat'].squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    bytes_est = int(bpp * h * w / 8)
    return Image.fromarray(recon), bytes_est


def main():
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    rows = []
    for fn in files:
        img = Image.open(os.path.join(KODAK_DIR, fn)).convert('RGB')
        arr = np.array(img)
        # CE-KMeans baseline
        out_ce = np.array(ce_kmeans(img, K))
        q = Image.fromarray(out_ce).convert('P', palette=Image.Palette.ADAPTIVE, colors=K)
        bio = io.BytesIO(); q.save(bio, format='PNG'); ce_bytes = bio.tell()
        ce_d = float(deltaE_ciede2000(rgb2lab(arr.astype(np.float64)/255.0),
                                      rgb2lab(out_ce.astype(np.float64)/255.0)).mean())
        ce_p = psnr(arr, out_ce)
        line = f'{fn}: CE dE00={ce_d:.3f} PSNR={ce_p:.1f} bytes={ce_bytes}'
        # Deep models
        for qidx in [2, 4, 6, 8]:
            try:
                out_d, bytes_d = compress_decompress(models[qidx], img)
                oa = np.array(out_d)
                d = float(deltaE_ciede2000(rgb2lab(arr.astype(np.float64)/255.0),
                                           rgb2lab(oa.astype(np.float64)/255.0)).mean())
                p = psnr(arr, oa)
                rows.append([fn, f'bmshj2018_q{qidx}', round(p, 2), round(d, 3), bytes_d])
                line += f' | q{qidx} dE00={d:.3f} PSNR={p:.1f} b={bytes_d}'
            except Exception as e:
                line += f' | q{qidx} FAILED'
                print(f'{fn} q{qidx} error: {e}')
        # CE-KMeans row
        rows.append([fn, 'CE-KMeans', round(ce_p, 2), round(ce_d, 3), ce_bytes])
        sys.stdout.write(line + '\n')
        sys.stdout.flush()

    # Save
    import csv
    with open(f'{_ROOT}/data/deep_comparison.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['image', 'method', 'psnr', 'dE00', 'bytes']); w.writerows(rows)
    print('\n=== Deep vs CE comparison (mean over 24) ===')
    import pandas as pd
    df = pd.read_csv(f'{_ROOT}/data/deep_comparison.csv')
    for m in df.method.unique():
        sub = df[df.method == m]
        print(f'{m:<20} PSNR={sub.psnr.mean():.1f} dE00={sub.dE00.mean():.3f} bytes={sub.bytes.mean():.0f}')


if __name__ == '__main__':
    main()