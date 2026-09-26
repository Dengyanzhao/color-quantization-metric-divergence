# Metric Divergence in Color Quantization

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
