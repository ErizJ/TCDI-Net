import argparse
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.nn as nn
import torch.utils.data as Data
import torch.utils.data.distributed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from dataset import (
    CVIU17Dataset,
    KonIQ10KDataset,
    LiveCDataset,
    NBUCIQADDataset,
    QADSDataset,
    RealSRQDataset,
    SISARDataset,
    Waterloo15Dataset,
)
from iqanet import TCDINet, SRIQALoss, ABLATION_CONFIGS, build_ablation_model

# ---------------------------------------------------------------------------
# Dataset registry: maps --dataset name to (DatasetClass, default_images_dir, default_mos_file)
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "cviu17":     (CVIU17Dataset,     "SRimages",              "mos_with_names_cviu17.csv"),
    "livesr":     (LiveCDataset,      "Imagesall",             "image_mos.csv"),
    "sisar":      (SISARDataset,      "SR",                    "MOS_with_names.csv"),
    "qads":       (QADSDataset,       "super-resolved_images", "mos_with_names.csv"),
    "waterloo15": (Waterloo15Dataset, "WIND_all",              "mos_with_names.csv"),
    # To be enabled in future releases:
    # "realsrq":    (RealSRQDataset,     "SR",                    "mos_with_names.csv"),
    # "koniq10k":   (KonIQ10KDataset,   "1024x768",              "koniq10k_scores_and_distributions.csv"),
    # "nbuciqad":   (NBUCIQADDataset,   "Cartoon_images",        "mos_with_names.csv"),
}


def get_args():
    parser = argparse.ArgumentParser(description="TCDI-Net: Texture-Structure Complementary Dual-branch Interaction Network for SR-IQA")

    # Dataset
    parser.add_argument("data", metavar="DIR", help="path to dataset root")
    parser.add_argument("--dataset", default="cviu17", choices=list(DATASET_CONFIG.keys()),
                        help="dataset name")
    parser.add_argument("--images-dir", default=None,
                        help="override subdirectory containing images")
    parser.add_argument("--mos-file", default=None,
                        help="override MOS CSV filename")
    parser.add_argument("--order-file", default=None,
                        help="override path to order file (train/test split)")

    # DR mode
    parser.add_argument("--dr-mode", action="store_true",
                        help="enable DR (Reduced-Reference) mode with LR images")
    parser.add_argument("--lr-dir", default=None,
                        help="directory containing LR/reference images (required for DR mode)")
    parser.add_argument("--no-pretrained-vgg", action="store_true",
                        help="disable pretrained VGG weights (use if torchvision is unavailable)")

    # Training
    parser.add_argument("--seed", default=42, type=int, help="random seed for reproducibility")
    parser.add_argument("-j", "--workers", default=2, type=int, help="number of data loading workers")
    parser.add_argument("--epochs", default=100, type=int, help="number of total epochs")
    parser.add_argument("-b", "--batch-size", default=4, type=int, help="mini-batch size")
    parser.add_argument("--lr", default=1e-1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, dest="weight_decay",
                        help="weight decay")
    parser.add_argument("--train-size", default=1300, type=int,
                        help="number of training samples (first N from order file)")
    parser.add_argument("--th", default=4, type=int,
                        help="which row of the order file to use (for k-fold splits)")

    # Distributed training
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:6328", type=str)
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--multiprocessing-distributed", action="store_true",
                        help="use multi-processing distributed training")

    # Evaluation
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate only")
    parser.add_argument("-p", "--pretrained", action="store_true", help="load pretrained model")
    parser.add_argument("-a", "--arch", default="checkpoint", help="checkpoint name (without suffix)")

    # Ablation
    parser.add_argument("--ablation", default=None, choices=list(ABLATION_CONFIGS.keys()),
                        help="ablation variant (overrides default full model)")

    # Logging
    parser.add_argument("--tensorboard", action="store_true", help="enable TensorBoard logging")
    parser.add_argument("--comment", default="", type=str, help="suffix for log and checkpoint names")

    args = parser.parse_args()
    args.timestep = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    # Resolve dataset configuration
    ds_cls, default_images_dir, default_mos_file = DATASET_CONFIG[args.dataset]
    args.dataset_cls = ds_cls
    args.images_dir = args.images_dir or default_images_dir
    args.mos_filename = args.mos_file or default_mos_file
    args.SR_images_folder = os.path.join(args.data, args.images_dir)
    args.mos_file = os.path.join(args.data, args.mos_filename)
    args.order_file = args.order_file or f"./data/orders/{args.dataset}_MOS_orders.csv"

    assert os.path.exists(args.SR_images_folder), f"images folder not found: {args.SR_images_folder}"
    assert os.path.exists(args.mos_file), f"MOS file not found: {args.mos_file}"
    assert os.path.exists(args.order_file), f"order file not found: {args.order_file}, run scripts/generate_order.py first"

    if args.dr_mode:
        assert args.lr_dir is not None, "--lr-dir is required when --dr-mode is enabled"
        args.lr_images_folder = os.path.join(args.data, args.lr_dir)
        assert os.path.exists(args.lr_images_folder), f"LR images folder not found: {args.lr_images_folder}"
    else:
        args.lr_images_folder = None

    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def main():
    args = get_args()
    set_seed(args.seed)

    if not os.path.exists("outputs/checkpoints"):
        os.makedirs("outputs/checkpoints")
    if not os.path.exists("result"):
        os.makedirs("result")

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join("runs", args.timestep + args.comment))

    if gpu is not None:
        print(f"Use GPU: {gpu} for training")

    if args.multiprocessing_distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=ngpus_per_node, rank=gpu)

    # Create model
    pretrained_vgg = not args.no_pretrained_vgg
    if args.pretrained:
        print("=> using pre-trained model")
        path = f"outputs/checkpoints/{args.arch}.pth.tar"
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        # Strip "module." prefix added by DataParallel
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        # Detect dr_mode from checkpoint config if available
        ckpt_dr_mode = checkpoint.get("args", None)
        if ckpt_dr_mode is not None and hasattr(ckpt_dr_mode, "dr_mode"):
            dr_mode = ckpt_dr_mode.dr_mode
        else:
            dr_mode = args.dr_mode
        model = TCDINet(dr_mode=dr_mode, pretrained_vgg=pretrained_vgg)
        model.load_state_dict(state_dict)
    elif args.ablation is not None:
        print(f"=> creating ablation model: {args.ablation} (gpu:{gpu})")
        model = build_ablation_model(args.ablation, dr_mode=args.dr_mode,
                                     pretrained_vgg=pretrained_vgg)
    else:
        print(f"=> creating model (gpu:{gpu})")
        model = TCDINet(dr_mode=args.dr_mode, pretrained_vgg=pretrained_vgg)

    device_ids = [0]
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()

    if args.multiprocessing_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = SRIQALoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)

    # Data loading
    mos_df = pd.read_csv(args.mos_file)
    order = pd.read_csv(args.order_file).iloc[args.th].to_numpy()
    order_train = order[:args.train_size]
    order_test = order[args.train_size:]
    mos_df_train = mos_df.iloc[order_train]
    mos_df_test = mos_df.iloc[order_test]

    train_dataset = args.dataset_cls(mos_df_train, args.SR_images_folder,
                                      lr_images_folder=args.lr_images_folder)
    val_dataset = args.dataset_cls(mos_df_test, args.SR_images_folder, training=False,
                                   lr_images_folder=args.lr_images_folder)

    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = Data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        pin_memory=True, num_workers=args.workers, drop_last=True,
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, 0, writer, args)
        return

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    metrics = {
        "train_PLCC": [], "train_SRCC": [], "train_RMSE": [], "train_Loss": [],
        "test_PLCC": [],  "test_SRCC": [],  "test_RMSE": [],  "test_Loss": [],
    }
    best_res = (0, 0, 0)

    for epoch in range(args.epochs):
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        PLCC, SRCC, RMSE, loss = train(train_loader, model, criterion, optimizer, epoch, writer, args)
        metrics["train_PLCC"].append(PLCC)
        metrics["train_SRCC"].append(SRCC)
        metrics["train_RMSE"].append(RMSE)
        metrics["train_Loss"].append(loss)

        loss, res = validate(val_loader, model, criterion, epoch, writer, args)
        metrics["test_PLCC"].append(res[0])
        metrics["test_SRCC"].append(res[1])
        metrics["test_RMSE"].append(res[2])
        metrics["test_Loss"].append(loss)

        scheduler.step(loss)

        if gpu == 0:
            is_best = res[0] > best_res[0]
            if is_best:
                best_res = res

            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_res": best_res,
                "args": args,
            }, is_best, f"outputs/checkpoints/{args.timestep}{args.comment}")

    # Save result CSV
    df = pd.DataFrame(metrics)
    result_path = f"result/{args.dataset}_results.csv"
    df.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")

    if writer:
        writer.flush()
        writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer, args):
    batch_time = AverageMeter("Time", ":6.2f", print_sum=True)
    data_time = AverageMeter("Data", ":6.2f", print_sum=True)
    losses = AverageMeter("Loss", ":.4e")
    result = ResultMeter()

    progress = ProgressMeter(args.epochs, [batch_time, data_time, losses, result], prefix="Train")

    model.train()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.dr_mode:
            images, lr_images, target = batch
            lr_images = lr_images.cuda(args.gpu, non_blocking=True)
        else:
            images, target = batch

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if args.dr_mode:
            output = model(images, lr_img=lr_images)
        else:
            output = model(images)
        loss = criterion(output, target)

        result.update(output, target)
        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(epoch, optimizer.param_groups)
    if args.tensorboard:
        writer_helper(writer, args.gpu, "train",
                      {"PLCC": result.PLCC, "SRCC": result.SRCC, "RMSE": result.RMSE, "LOSS": losses.avg}, epoch)

    return result.PLCC, result.SRCC, result.RMSE, losses.avg


def validate(val_loader, model, criterion, epoch, writer, args):
    batch_time = AverageMeter("Time", ":6.2f", print_sum=True)
    data_time = AverageMeter("Data", ":6.2f", print_sum=True)
    losses = AverageMeter("Loss", ":.4e")
    result = ResultMeter()

    progress = ProgressMeter(args.epochs, [batch_time, data_time, losses, result], prefix="*Test")

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            data_time.update(time.time() - end)

            if args.dr_mode:
                images, lr_images, target = batch
                lr_images = lr_images.cuda(args.gpu, non_blocking=True)
            else:
                images, target = batch

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            if args.dr_mode:
                output = model(images, lr_img=lr_images)
            else:
                output = model(images)
            loss = criterion(output, target)

            result.update(output, target)
            losses.update(loss.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

    progress.display(epoch)
    if args.tensorboard:
        writer_helper(writer, args.gpu, "valid",
                      {"PLCC": result.PLCC, "SRCC": result.SRCC, "RMSE": result.RMSE, "LOSS": losses.avg}, epoch)

    return losses.avg, (result.PLCC, result.SRCC, result.RMSE)


def writer_helper(writer, gpu, tag, to_tensorboard, epoch):
    writer.add_scalars(f"gpu{gpu}/PLCC", {tag: to_tensorboard["PLCC"]}, global_step=epoch)
    writer.add_scalars(f"gpu{gpu}/SRCC", {tag: to_tensorboard["SRCC"]}, global_step=epoch)
    writer.add_scalars(f"gpu{gpu}/RMSE", {tag: to_tensorboard["RMSE"]}, global_step=epoch)
    writer.add_scalars(f"gpu{gpu}/LOSS", {tag: to_tensorboard["LOSS"]}, global_step=epoch)


def score_to_mos(t):
    """Extract scalar MOS from model output tensor of shape (N, 1)."""
    assert t.shape[1] == 1
    return t[:, 0].detach().numpy()


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + ".pth.tar")
    if is_best:
        shutil.copyfile(filename + ".pth.tar", filename + "_best.pth.tar")


# ---------------------------------------------------------------------------
# Utility classes
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self, name, fmt=":f", print_sum=False):
        self.name = name
        self.fmt = fmt
        self.print_sum = print_sum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.print_sum:
            return f"{self.name} {self.sum:{self.fmt[1:]}}"
        return f"{self.name} {self.avg:{self.fmt[1:]}}"


class ResultMeter:
    def __init__(self):
        self.y_pred_all = torch.zeros(0, dtype=torch.float64)
        self.y_all = torch.zeros(0, dtype=torch.float64)
        self.PLCC = 0
        self.SRCC = 0
        self.RMSE = 0

    def update(self, y_pred, y):
        self.y_pred_all = torch.cat((self.y_pred_all, y_pred.cpu()), dim=0)
        self.y_all = torch.cat((self.y_all, y.cpu()), dim=0)

    def __str__(self):
        y_all = score_to_mos(self.y_all)
        y_pred_all = score_to_mos(self.y_pred_all)
        self.PLCC = pearsonr(y_all, y_pred_all)[0]
        self.SRCC = spearmanr(y_all, y_pred_all)[0]
        self.RMSE = np.sqrt(mean_squared_error(y_all, y_pred_all))
        return f"PLCC={self.PLCC:.4f}|SRCC={self.SRCC:.4f}|RMSE={self.RMSE:.4f}"


class ProgressMeter:
    def __init__(self, num_epochs, meters, prefix=""):
        self.epoch_fmtstr = self._get_epoch_fmtstr(num_epochs)
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch, param_groups=None):
        entries = [self.prefix, time.strftime("%m-%d %H:%M:%S", time.localtime()),
                   self.epoch_fmtstr.format(epoch)]
        entries += [str(meter) for meter in self.meters]
        if param_groups is not None:
            entries += ["(lr:{})".format("/".join([f"{p['lr']:.0e}" for p in param_groups]))]
        print(" ".join(entries))

    def _get_epoch_fmtstr(self, num_epochs):
        num_digits = len(str(num_epochs // 1))
        return "[" + "{:" + str(num_digits) + "d}" + "/" + str(num_epochs) + "]"


if __name__ == "__main__":
    main()
