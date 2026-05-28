import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


class KonIQ10KDataset(Dataset):
    """KonIQ-10k dataset."""

    def __init__(self, mos_df, images_folder, training=True, dist=True, lr_images_folder=None):
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.distribution = dist
        self.lr_images_folder = lr_images_folder

        if training:
            if lr_images_folder is not None:
                self.transforms = transforms.Compose([
                    transforms.CenterCrop((768, 1024)),
                    transforms.ToTensor(),
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(3, expand=True),
                    transforms.CenterCrop((768, 1024)),
                    transforms.ToTensor(),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop((768, 1024)),
                transforms.ToTensor(),
            ])

        self.images_folder = images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        image_path = os.path.join(self.images_folder, mos_detail.image_name)
        image = self.transforms(Image.open(image_path).convert("RGB"))

        if self.distribution:
            mos_distribution = (mos_detail.c1, mos_detail.c2,
                                mos_detail.c3, mos_detail.c4, mos_detail.c5)
            label = tuple([m / mos_detail.c_total for m in mos_distribution])
        else:
            label = [mos_detail.MOS / 5]
        label = torch.Tensor(label)

        if self.lr_images_folder is not None:
            lr_filename = os.path.basename(mos_detail.image_name)
            lr_path = os.path.join(self.lr_images_folder, lr_filename)
            lr_image = self.transforms(Image.open(lr_path).convert("RGB"))
            return image, lr_image, label

        return image, label


class LiveCDataset(Dataset):
    """Live-C dataset."""

    def __init__(self, mos_df, images_folder, training=True, lr_images_folder=None):
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.lr_images_folder = lr_images_folder

        if training:
            if lr_images_folder is not None:
                self.transforms = transforms.Compose([
                    transforms.CenterCrop((318, 318)),
                    transforms.ToTensor(),
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomCrop((318, 318)),
                    transforms.ToTensor(),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop((318, 318)),
                transforms.ToTensor(),
            ])

        self.images_folder = images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        image_path = os.path.join(self.images_folder, mos_detail.ImageName)
        image = self.transforms(Image.open(image_path).convert("RGB"))
        label = torch.Tensor([mos_detail.MOS / 100])

        if self.lr_images_folder is not None:
            lr_filename = os.path.basename(mos_detail.ImageName)
            lr_path = os.path.join(self.lr_images_folder, lr_filename)
            lr_image = self.transforms(Image.open(lr_path).convert("RGB"))
            return image, lr_image, label

        return image, label


class CVIU17Dataset(Dataset):
    """CVIU17 dataset."""

    def __init__(self, mos_df, SR_images_folder, training=True, lr_images_folder=None):
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.lr_images_folder = lr_images_folder

        if training:
            if lr_images_folder is not None:
                self.transforms = transforms.Compose([
                    transforms.CenterCrop((318, 318)),
                    transforms.ToTensor(),
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.RandomCrop((318, 318)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop((318, 318)),
                transforms.ToTensor(),
            ])

        self.SR_images_folder = SR_images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        SR_image_path = os.path.join(self.SR_images_folder, mos_detail.SR_image_name)
        SR_image = self.transforms(Image.open(SR_image_path).convert("RGB"))
        label = torch.Tensor([mos_detail.MOS / 10])

        if self.lr_images_folder is not None:
            lr_filename = os.path.basename(mos_detail.SR_image_name)
            lr_path = os.path.join(self.lr_images_folder, lr_filename)
            lr_image = self.transforms(Image.open(lr_path).convert("RGB"))
            return SR_image, lr_image, label

        return SR_image, label


class RealSRQDataset(Dataset):
    """RealSRQ dataset."""

    def __init__(self, mos_df, SR_images_folder, training=True, lr_images_folder=None):
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.lr_images_folder = lr_images_folder

        if training:
            if lr_images_folder is not None:
                self.transforms = transforms.Compose([
                    transforms.CenterCrop((318, 318)),
                    transforms.ToTensor(),
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomCrop((318, 318)),
                    transforms.ToTensor(),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop((318, 318)),
                transforms.ToTensor(),
            ])

        self.SR_images_folder = SR_images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        SR_image_path = os.path.join(self.SR_images_folder, str(mos_detail.SR_image))
        SR_image = self.transforms(Image.open(SR_image_path).convert("RGB"))
        label = torch.Tensor([mos_detail.mos])

        if self.lr_images_folder is not None:
            lr_filename = os.path.basename(str(mos_detail.SR_image))
            lr_path = os.path.join(self.lr_images_folder, lr_filename)
            lr_image = self.transforms(Image.open(lr_path).convert("RGB"))
            return SR_image, lr_image, label

        return SR_image, label


class NBUCIQADDataset(Dataset):
    """NBU-CIQAD dataset (MCCIs / SCCIs)."""

    def __init__(self, mos_df, Cartoon_images_folder, training=True, lr_images_folder=None):
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.lr_images_folder = lr_images_folder

        if training:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.Cartoon_images_folder = Cartoon_images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        Cartoon_image_path = os.path.join(self.Cartoon_images_folder, mos_detail.cartoon_image)
        Cartoon_image = self.transforms(Image.open(Cartoon_image_path).convert("RGB"))
        label = torch.Tensor([mos_detail.mos / 100])

        if self.lr_images_folder is not None:
            lr_filename = os.path.basename(mos_detail.cartoon_image)
            lr_path = os.path.join(self.lr_images_folder, lr_filename)
            lr_image = self.transforms(Image.open(lr_path).convert("RGB"))
            return Cartoon_image, lr_image, label

        return Cartoon_image, label


class SISARDataset(Dataset):
    """SISAR dataset."""

    def __init__(self, mos_df, SR_images_folder, training=True, lr_images_folder=None):
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.lr_images_folder = lr_images_folder

        if training:
            if lr_images_folder is not None:
                self.transforms = transforms.Compose([
                    transforms.CenterCrop((318, 318)),
                    transforms.ToTensor(),
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomCrop((318, 318)),
                    transforms.ToTensor(),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop((318, 318)),
                transforms.ToTensor(),
            ])

        self.SR_images_folder = SR_images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        SR_image_path = os.path.join(self.SR_images_folder, mos_detail.SR_images)
        SR_image = self.transforms(Image.open(SR_image_path).convert("RGB"))
        label = torch.Tensor([mos_detail.MOS])

        if self.lr_images_folder is not None:
            lr_filename = os.path.basename(mos_detail.SR_images)
            lr_path = os.path.join(self.lr_images_folder, lr_filename)
            lr_image = self.transforms(Image.open(lr_path).convert("RGB"))
            return SR_image, lr_image, label

        return SR_image, label


class QADSDataset(Dataset):
    """QADS dataset."""

    def __init__(self, mos_df, SR_images_folder, training=True, lr_images_folder=None):
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.lr_images_folder = lr_images_folder

        if training:
            if lr_images_folder is not None:
                self.transforms = transforms.Compose([
                    transforms.CenterCrop((318, 318)),
                    transforms.ToTensor(),
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.CenterCrop((318, 318)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop((318, 318)),
                transforms.ToTensor(),
            ])

        self.SR_images_folder = SR_images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        SR_image_path = os.path.join(self.SR_images_folder, mos_detail.sri_image_name)
        SR_image = self.transforms(Image.open(SR_image_path).convert("RGB"))
        label = torch.Tensor([mos_detail.MOS])

        if self.lr_images_folder is not None:
            lr_filename = os.path.basename(mos_detail.sri_image_name)
            lr_path = os.path.join(self.lr_images_folder, lr_filename)
            lr_image = self.transforms(Image.open(lr_path).convert("RGB"))
            return SR_image, lr_image, label

        return SR_image, label


class Waterloo15Dataset(Dataset):
    """Waterloo15 / WIND dataset."""

    def __init__(self, mos_df, SR_images_folder, training=True, lr_images_folder=None):
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.lr_images_folder = lr_images_folder

        if training:
            if lr_images_folder is not None:
                self.transforms = transforms.Compose([
                    transforms.CenterCrop((318, 318)),
                    transforms.ToTensor(),
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.CenterCrop((318, 318)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop((318, 318)),
                transforms.ToTensor(),
            ])

        self.SR_images_folder = SR_images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        SR_image_path = os.path.join(self.SR_images_folder, mos_detail.SR_Images)
        SR_image = self.transforms(Image.open(SR_image_path).convert("RGB"))
        label = torch.Tensor([mos_detail.MOS / 10])

        if self.lr_images_folder is not None:
            lr_filename = os.path.basename(mos_detail.SR_Images)
            lr_path = os.path.join(self.lr_images_folder, lr_filename)
            lr_image = self.transforms(Image.open(lr_path).convert("RGB"))
            return SR_image, lr_image, label

        return SR_image, label


class LiveDataSet(Dataset):
    """Live-itW dataset for cross-dataset evaluation (full-resolution, no crop)."""

    def __init__(self, mos_df, images_folder, lr_images_folder=None):
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.lr_images_folder = lr_images_folder

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.images_folder = images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        image_path = os.path.join(self.images_folder, mos_detail[0])
        image = self.transforms(Image.open(image_path).convert("RGB"))
        label = torch.Tensor([mos_detail[1] / 20])

        if self.lr_images_folder is not None:
            lr_filename = os.path.basename(mos_detail[0])
            lr_path = os.path.join(self.lr_images_folder, lr_filename)
            lr_image = self.transforms(Image.open(lr_path).convert("RGB"))
            return image, lr_image, label

        return image, label
