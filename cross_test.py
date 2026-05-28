"""Cross-dataset evaluation script for TCDI-Net.

Evaluates a trained model on the Live-itW dataset.
Usage:
    python cross_test.py /path/to/LiveChallenge <checkpoint_name>
    python cross_test.py /path/to/LiveChallenge <checkpoint_name> --dr-mode --lr-dir LR
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from dataset import LiveDataSet
from iqanet import TCDINet


def score_to_mos(t):
    """Extract scalar MOS from model output tensor of shape (N, 1)."""
    return t[:, 0].detach().cpu().numpy()


def validate(val_loader, model, dr_mode=False):
    result = ResultMeter()
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if dr_mode:
                images, lr_images, target = batch
                lr_images = lr_images.cuda(non_blocking=True)
            else:
                images, target = batch

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            if dr_mode:
                output = model(images, lr_img=lr_images)
            else:
                output = model(images)
            result.update(output, target)
    print(f"*Test {result}")


class ResultMeter:
    def __init__(self):
        self.y_pred_all = torch.zeros(0, dtype=torch.float32)
        self.y_all = torch.zeros(0, dtype=torch.float32)

    def update(self, y_pred, y):
        self.y_pred_all = torch.cat((self.y_pred_all, y_pred.cpu()), dim=0)
        self.y_all = torch.cat((self.y_all, y.cpu()), dim=0)

    def __str__(self):
        y_all = score_to_mos(self.y_all)
        y_pred_all = score_to_mos(self.y_pred_all)
        PLCC = pearsonr(y_all, y_pred_all)[0]
        SRCC = spearmanr(y_all, y_pred_all)[0]
        RMSE = np.sqrt(mean_squared_error(y_all, y_pred_all))
        return f"PLCC={PLCC:.4f}|SRCC={SRCC:.4f}|RMSE={RMSE:.4f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation on Live-itW")
    parser.add_argument("data", metavar="DIR", help="path to LiveChallenge dataset root")
    parser.add_argument("checkpoint", metavar="CHECKPOINT", help="checkpoint name (without .pth.tar suffix)")
    parser.add_argument("--dr-mode", action="store_true", help="enable DR (Reduced-Reference) mode")
    parser.add_argument("--lr-dir", default=None, help="directory containing LR images (required for DR mode)")
    args = parser.parse_args()

    images_folder = os.path.join(args.data, "Images")
    assert os.path.exists(images_folder), f"Live-itW images folder not found: {images_folder}"

    mos_file = os.path.join(args.data, "Data", "live_moc.csv")
    assert os.path.exists(mos_file), f"Live-itW MOS file not found: {mos_file}"
    mos = pd.read_csv(mos_file, header=None)

    checkpoint_path = f"outputs/checkpoints/{args.checkpoint}.pth.tar"
    assert os.path.exists(checkpoint_path), f"checkpoint not found: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    # Strip "module." prefix added by DataParallel
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    ckpt_args = checkpoint.get("args", None)
    dr_mode = args.dr_mode
    if ckpt_args is not None and hasattr(ckpt_args, "dr_mode"):
        dr_mode = ckpt_args.dr_mode or args.dr_mode

    model = TCDINet(dr_mode=dr_mode)
    model.load_state_dict(state_dict)
    model.cuda()

    lr_folder = None
    if dr_mode:
        assert args.lr_dir is not None, "--lr-dir is required when --dr-mode is enabled"
        lr_folder = os.path.join(args.data, args.lr_dir)
        assert os.path.exists(lr_folder), f"LR images folder not found: {lr_folder}"

    loader = Data.DataLoader(
        dataset=LiveDataSet(mos, images_folder, lr_images_folder=lr_folder),
        batch_size=1, pin_memory=True,
    )
    validate(loader, model, dr_mode=dr_mode)
