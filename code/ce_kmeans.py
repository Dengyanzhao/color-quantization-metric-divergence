# -*- coding: utf-8 -*-
"""CE-KMeans: CIEDE2000-Enhanced K-means color quantization.
Assignment step uses CIEDE2000-structured weighted distance in LCh space
(SL/SC/SH scale factors from the CIEDE2000 formula, recomputed per iteration
from current cluster centers). Update step = weighted mean in Lab space.
Also implements iterative palette-size search (no training needed)."""

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
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import time, csv, os, sys, io
from numba import njit, prange
from math import sqrt, sin, cos, exp, atan2
PI = 3.141592653589793
from skimage.metrics import structural_similarity as ssim_fn
from skimage.color import deltaE_ciede2000

RNG = np.random.default_rng(42)
KODAK_DIR = f'{_DATASETS}/kodak'
OUT_CSV = f'{_ROOT}/data/ce_results.csv'
N_COLORS = [16, 32, 64, 128, 256]
SAMPLE = 50000
MAX_ITER = 20


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def lab2lch(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(a ** 2 + b ** 2)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([L, C, h], axis=-1)


def ciede2000_weights(centers_lch):
    """SL/SC/SH per center from CIEDE2000 (reference color = center)."""
    L, C, h = centers_lch[:, 0], centers_lch[:, 1], centers_lch[:, 2]
    SL = 1.0 + 0.015 * (L - 50.0) ** 2 / np.sqrt(20.0 + (L - 50.0) ** 2)
    SC = 1.0 + 0.045 * C
    T = (1.0 - 0.17 * np.cos(np.radians(h - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * h))
         + 0.32 * np.cos(np.radians(3.0 * h + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * h - 63.0)))
    SH = 1.0 + 0.015 * C * T
    return SL, SC, SH


def assign_ce(lab_pix, centers_lab, chunk=15000):
    """Assignment with the complete symmetric CIEDE2000 distance
    (G-factor chroma correction, mean L/C/h, R_T rotation term).
    Delegates to the numba-accelerated full implementation."""
    del chunk  # 完整版一次性计算
    labels = assign_ce_full_njit(np.ascontiguousarray(lab_pix, dtype=np.float64),
                                 np.ascontiguousarray(centers_lab, dtype=np.float64))
    return labels.astype(np.int32)


def ce_kmeans(img, n_colors, max_iter=MAX_ITER, lam=0.0):
    """CE-KMeans: CIEDE2000-guided clustering, median-cut init."""
    arr = np.array(img).astype(np.float64) / 255.0
    lab = rgb2lab(arr)
    h, w, _ = lab.shape
    pixels = lab.reshape(-1, 3).astype(np.float32)

    mc = img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    centers_rgb = np.array(mc.getpalette()[:n_colors * 3]).reshape(-1, 3).astype(np.float64)
    centers = rgb2lab(centers_rgb[None, :, :] / 255.0)[0].astype(np.float32)

    # RGB 数据（用于 λ 混合距离）
    arr_rgb = (arr * 255.0).astype(np.float32)
    pixels_rgb_f = arr_rgb.reshape(-1, 3)
    centers_rgb_f = centers_rgb.astype(np.float32)

    idx = RNG.choice(len(pixels), min(SAMPLE, len(pixels)), replace=False)
    tr = pixels[idx]
    tr_rgb = pixels_rgb_f[idx]
    tr_lch = lab2lch(tr).astype(np.float32)

    for _ in range(max_iter):
        if lam > 0.0:
            c_lch = lab2lch(centers)
            SL, SC, SH = ciede2000_weights(c_lch)
            labels = assign_ce_lambda_njit(tr, c_lch, tr_lch,
                                           SL.astype(np.float32), SC.astype(np.float32), SH.astype(np.float32),
                                           tr_rgb, centers_rgb_f, lam)
        else:
            labels = assign_ce_full_njit(tr.astype(np.float64), centers.astype(np.float64))
        new_centers = np.zeros_like(centers)
        for k in range(n_colors):
            m = (labels == k)
            if m.sum() > 0:
                new_centers[k] = tr[m].mean(axis=0)
            else:
                new_centers[k] = centers[k]
        if np.allclose(centers, new_centers, atol=1e-3):
            break
        centers = new_centers

    if lam > 0.0:
        p_lch = lab2lch(pixels).astype(np.float32)
        c_lch = lab2lch(centers)
        SL, SC, SH = ciede2000_weights(c_lch)
        labels = assign_ce_lambda_njit(pixels, c_lch, p_lch,
                                       SL.astype(np.float32), SC.astype(np.float32), SH.astype(np.float32),
                                       pixels_rgb_f, centers_rgb_f, lam)
    else:
        labels = assign_ce_full_njit(pixels.astype(np.float64), centers.astype(np.float64))
    centers_out = lab2rgb(centers[None, ...])[0].clip(0, 1)
    out = (centers_out * 255).astype(np.uint8)[labels].reshape(h, w, 3)
    return Image.fromarray(out)


def iterative_palette(img, tau, k_start=16, k_max=256):
    """Training-free iterative palette size search.
    Start at k_start, double until mean CIEDE2000 <= tau or k_max reached.
    Returns (K, quantized image, mean dE00)."""
    arr = np.array(img)
    lab_ref = rgb2lab(arr.astype(np.float64) / 255.0)
    k = k_start
    best = None
    while k <= k_max:
        out = ce_kmeans(img, k)
        d = float(deltaE_ciede2000(lab_ref, rgb2lab(np.array(out).astype(np.float64) / 255.0)).mean())
        if best is None or d <= tau:
            best = (k, out, d)
            if d <= tau:
                break
        k *= 2
    return best


def run_one(img_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    rows = []
    for n in N_COLORS:
        t0 = time.perf_counter()
        out = ce_kmeans(img, n)
        dt = (time.perf_counter() - t0) * 1000.0
        out_arr = np.array(out)
        p = psnr(arr, out_arr)
        s = float(ssim_fn(arr, out_arr, channel_axis=2, data_range=255))
        d = float(deltaE_ciede2000(rgb2lab(arr.astype(np.float64)/255.0),
                                   rgb2lab(out_arr.astype(np.float64)/255.0)).mean())
        buf = out.convert('P', palette=Image.Palette.ADAPTIVE, colors=n)
        bio = io.BytesIO()
        buf.save(bio, format='PNG')
        rows.append([os.path.basename(img_path), 'ce_kmeans', n,
                     round(p, 3), round(float(s), 4), round(dt, 1), bio.tell(), round(d, 3)])
        sys.stdout.write(f"{os.path.basename(img_path)} ce_kmeans n={n}: "
                         f"PSNR={p:.2f} SSIM={s:.4f} dE00={d:.3f} t={dt:.0f}ms\n")
        sys.stdout.flush()
    return rows


def main():
    files = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'method', 'n_colors', 'psnr', 'ssim', 'time_ms', 'png_bytes', 'dE00'])
        for fn in files:
            w.writerows(run_one(os.path.join(KODAK_DIR, fn)))
            f.flush()
    print('DONE. CE-KMeans results in', OUT_CSV)


@njit(cache=True, parallel=True)
def assign_ce_full_njit(pixels_lab, centers_lab):
    """完整对称 CIEDE2000 分配（含 G 因子色度修正、平均 L/C/h、R_T 旋转项）。
    pixels_lab: (N,3) float64 Lab, centers_lab: (K,3) float64 Lab.
    返回每个像素最近中心的索引（完整 CIEDE2000 距离，与 skimage deltaE_ciede2000 一致）。"""
    n = pixels_lab.shape[0]
    k = centers_lab.shape[0]
    labels = np.empty(n, dtype=np.int64)
    for i in prange(n):
        L1 = pixels_lab[i, 0]
        a1 = pixels_lab[i, 1]
        b1 = pixels_lab[i, 2]
        C1 = sqrt(a1*a1 + b1*b1)
        best = 1e30
        bk = 0
        for j in range(k):
            L2 = centers_lab[j, 0]
            a2 = centers_lab[j, 1]
            b2 = centers_lab[j, 2]
            C2 = sqrt(a2*a2 + b2*b2)
            # G 因子（a' 色度修正）
            Cbar = 0.5 * (C1 + C2)
            G = 0.5 * (1.0 - sqrt(Cbar**7.0 / (Cbar**7.0 + 25.0**7.0)))
            a1p = (1.0 + G) * a1
            a2p = (1.0 + G) * a2
            C1p = sqrt(a1p*a1p + b1*b1)
            C2p = sqrt(a2p*a2p + b2*b2)
            # 色相角（度）
            h1p = atan2(b1, a1p) * 180.0 / PI
            h2p = atan2(b2, a2p) * 180.0 / PI
            if h1p < 0.0:
                h1p += 360.0
            if h2p < 0.0:
                h2p += 360.0
            # 差值
            dLp = L2 - L1
            dCp = C2p - C1p
            # 平均色相（跨 0° 处理）
            mean_L = 0.5 * (L1 + L2)
            mean_C = 0.5 * (C1p + C2p)
            hdiff = h2p - h1p
            avg_h = 0.5 * (h1p + h2p)
            if abs(hdiff) > 180.0:
                avg_h += 180.0
            if avg_h < 0.0:
                avg_h += 360.0
            elif avg_h >= 360.0:
                avg_h -= 360.0
            # C1p*C2p=0 时 h 无定义，用和
            if C1p * C2p == 0.0:
                avg_h = h1p + h2p
            # Δh（带符号，跨 0° 修正）
            dh = h2p - h1p
            if dh > 180.0:
                dh -= 360.0
            elif dh < -180.0:
                dh += 360.0
            if C1p * C2p == 0.0:
                dh = 0.0
            # ΔH'
            dHp = 2.0 * sqrt(C1p * C2p) * sin(dh * 0.5 * PI / 180.0)
            # SL/SC/SH
            SL = 1.0 + 0.015 * (mean_L - 50.0)**2.0 / sqrt(20.0 + (mean_L - 50.0)**2.0)
            SC = 1.0 + 0.045 * mean_C
            T = (1.0 - 0.17*cos((avg_h - 30.0) * PI / 180.0)
                 + 0.24*cos(2.0*avg_h * PI / 180.0)
                 + 0.32*cos((3.0*avg_h + 6.0) * PI / 180.0)
                 - 0.20*cos((4.0*avg_h - 63.0) * PI / 180.0))
            SH = 1.0 + 0.015 * mean_C * T
            # R_T 旋转项
            dtheta = 30.0 * exp(-((avg_h - 275.0) / 25.0)**2.0)
            RC = 2.0 * sqrt(mean_C**7.0 / (mean_C**7.0 + 25.0**7.0))
            RT = -sin(2.0*dtheta * PI / 180.0) * RC
            # 最终距离
            dL_term = (dLp / SL)**2.0
            dC_term = (dCp / SC)**2.0
            dH_term = (dHp / SH)**2.0
            dR_term = RT * (dCp / SC) * (dHp / SH)
            d2 = dL_term + dC_term + dH_term + dR_term
            if d2 < best:
                best = d2
                bk = j
        labels[i] = bk
    return labels


@njit(cache=True, parallel=True)
def assign_ce_njit(pixels_lab, centers_lch, p_lch_cache, center_SL, center_SC, center_SH):
    """Numba 加速的 CIEDE2000 分配（pixels_lab: (N,3), centers_lch: (K,3)）"""
    n = pixels_lab.shape[0]
    k = centers_lch.shape[0]
    labels = np.empty(n, dtype=np.int32)
    # 预计算像素 LCh 已由外部传入 p_lch_cache (N,3)
    for i in prange(n):
        Lp = p_lch_cache[i, 0]
        Cp = p_lch_cache[i, 1]
        hp = p_lch_cache[i, 2]
        best = 1e30
        bk = 0
        for j in range(k):
            dL = (Lp - centers_lch[j, 0]) / center_SL[j]
            dC = (Cp - centers_lch[j, 1]) / center_SC[j]
            dh = hp - centers_lch[j, 2]
            if dh > 180.0:
                dh -= 360.0
            if dh < -180.0:
                dh += 360.0
            dH = 2.0 * (Cp * centers_lch[j, 1]) ** 0.5 * np.sin(dh * 0.5 * 0.017453292519943295) / center_SH[j]
            # R_T rotation term (CIEDE2000 full formula)
            Cbar = 0.5 * (Cp + centers_lch[j, 1])
            hbar = 0.5 * (hp + centers_lch[j, 2])
            RC = 2.0 * (Cbar ** 7.0 / (Cbar ** 7.0 + 25.0 ** 7.0)) ** 0.5
            dtheta = 30.0 * np.exp(-((hbar - 275.0) / 25.0) ** 2.0)
            RT = -np.sin(2.0 * dtheta * 0.017453292519943295) * RC
            d2 = dL * dL + dC * dC + dH * dH + RT * dC * dH
            if d2 < best:
                best = d2
                bk = j
        labels[i] = bk
    return labels


@njit(cache=True, parallel=True)
def assign_ce_lambda_njit(pixels_lab, centers_lch, p_lch_cache, center_SL, center_SC, center_SH,
                          pixels_rgb, centers_rgb, lam):
    """CE-KMeans with lambda-blended distance: d = (1-lam)*d_CE + lam*(d_RGB/441)."""
    n = pixels_lab.shape[0]
    k = centers_lch.shape[0]
    labels = np.empty(n, dtype=np.int32)
    for i in range(n):
        Lp = p_lch_cache[i, 0]
        Cp = p_lch_cache[i, 1]
        hp = p_lch_cache[i, 2]
        rp = pixels_rgb[i, 0]; gp = pixels_rgb[i, 1]; bp = pixels_rgb[i, 2]
        best = 1e30
        bk = 0
        for j in range(k):
            dL = (Lp - centers_lch[j, 0]) / center_SL[j]
            dC = (Cp - centers_lch[j, 1]) / center_SC[j]
            dh = hp - centers_lch[j, 2]
            if dh > 180.0:
                dh -= 360.0
            if dh < -180.0:
                dh += 360.0
            dH = 2.0 * (Cp * centers_lch[j, 1]) ** 0.5 * np.sin(dh * 0.5 * 0.017453292519943295) / center_SH[j]
            d2 = dL * dL + dC * dC + dH * dH
            d_ce = d2 ** 0.5
            d_rgb = ((rp - centers_rgb[j, 0]) ** 2 + (gp - centers_rgb[j, 1]) ** 2 + (bp - centers_rgb[j, 2]) ** 2) ** 0.5
            d_mix = (1.0 - lam) * d_ce + lam * (d_rgb / 441.0)
            if d_mix < best:
                best = d_mix
                bk = j
        labels[i] = bk
    return labels

if __name__ == '__main__':
    main()
