# TCDI-Net

## Beyond Uniform Evaluation: Texture-Conditioned Complementary Decomposition and Interaction for Super-Resolution Image Quality Assessment

> **Accepted / In press** — *Digital Signal Processing* (Elsevier)
>
> JCR Impact Factor: **3.0** | JCR Q2 (Engineering, Electrical & Electronic) | 中科院大类 **2区** / 小类 **1区**

Official implementation of TCDI-Net, a texture-conditioned complementary decomposition and interaction framework for super-resolution image quality assessment (SR-IQA). The method supports both **No-Reference (NR)** and **Degraded-Reference (DR)** quality prediction by dynamically modeling structural and textural cues in super-resolved images.

## Architecture Overview

TCDI-Net consists of two complementary branches:

- **Image Branch**: Extracts multi-scale features from the SR image via a ResBlock + DCAB backbone, progressively refined by texture-guided feedback (TGB).
- **Detail Branch**: Processes the texture/detail component $I_T$ obtained through scale-controlled Gaussian decomposition, providing conditioning signals for both TGB and the Gated Dynamic Interaction Block (GDIB).

Key components:

| Component | Description |
|-----------|-------------|
| Gaussian Decomposition | Input-level separation of base ($I_S$) and detail ($I_T$) components |
| TGB (Texture Guided Block) | Detail-conditioned spatial reweighting at each image branch stage |
| GDIB (Gated Dynamic Interaction Block) | Dynamic low/high-frequency decomposition conditioned on texture features |
| MSFFB (Multi-Scale Feature Fusion Block) | Cross-scale fusion with channel-spatial attention |
| Structure Compensation Branch | (DR mode) Weak-reference comparison using frozen VGG-16 features |

### Modes

| Mode | Input | Description |
|------|-------|-------------|
| NR (`dr_mode=False`) | SR image only | No-Reference quality prediction |
| DR (`dr_mode=True`) | SR image + LR image | Degraded-Reference with structure compensation |

## Requirements

```bash
pip install -r requirements.txt
```

- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- numpy, pandas, scipy, scikit-learn, Pillow

## Quick Start

```python
import torch
from iqanet import TCDINet

# NR mode
model = TCDINet(dr_mode=False)
score = model(torch.rand(1, 3, 256, 256))

# DR mode (with LR reference)
model_dr = TCDINet(dr_mode=True)
score_dr = model_dr(
    torch.rand(1, 3, 256, 256),      # SR image
    lr_img=torch.rand(1, 3, 64, 64)  # LR image
)
```

## Training

```bash
# NR mode
python main.py /path/to/dataset --dataset cviu17 --epochs 100 --batch-size 4

# DR mode
python main.py /path/to/dataset --dataset cviu17 --dr-mode --lr-dir LRimages --epochs 100
```

## Evaluation

```bash
python main.py /path/to/dataset --dataset cviu17 --evaluate --pretrained --arch checkpoint_name
python cross_test.py /path/to/LiveChallenge checkpoint_name
```

## Ablation Study

```bash
python main.py /path/to/dataset --dataset cviu17 --ablation baseline --epochs 100
python ablation.py /path/to/dataset --dataset cviu17 --epochs 100
```

Available variants: `baseline`, `tgb`, `msffb`, `dcab`, `bop`, `gdib`, `tgb+msffb`, `tgb+msffb+dcab`, `tgb+msffb+dcab+bop`, `full`.

## Datasets

| `--dataset` | Dataset |
|-------------|---------|
| `cviu17` | CVIU17 |
| `livesr` | LIVE-SR |
| `sisar` | SISAR |
| `qads` | QADS |
| `waterloo15` | Waterloo15 (WIND) |
| `realsrq` | RealSRQ *(coming soon)* |

> **Coming soon**: KonIQ-10k, NBU-CIQAD

## Citation

If you find this work useful, please cite:

```bibtex
@article{tcdinet,
  title   = {Beyond Uniform Evaluation: {T}exture-Conditioned Complementary Decomposition and Interaction for Super-Resolution Image Quality Assessment},
  journal = {Digital Signal Processing},
  year    = {2025},
  note    = {Accepted / In press}
}
```

## License

This project is released under the [MIT License](LICENSE).
