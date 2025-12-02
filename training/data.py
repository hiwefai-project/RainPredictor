import os
import glob
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms

import rasterio
from rasterio.errors import RasterioIOError

import torchio as tio

from .config import PRED_LENGTH, PIN_MEMORY
from .utils import set_seed


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize radar reflectivity (dBZ) image to [0, 1] range."""
    # Line that 'clear' the radar image from nan, +inf, -inf values.
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    # Set lower bound. If there are values lower than 0, will be set to 0.
    img = np.clip(img, 0, None)
    min_dbz, max_dbz = 0, 70
    # Set upper and lower bound.
    img = np.clip(img, min_dbz, max_dbz)
    # Normalize image between 0 and 1
    return (img - min_dbz) / (max_dbz - min_dbz)


def get_augmentation_transforms() -> tio.Compose:
    """Build TorchIO-based augmentation pipeline for spatio-temporal radar volumes."""
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1), p=0.5),
        tio.RandomAffine(scales=(0.8, 1.2), degrees=90, p=0.5),
    ])


def tiff_is_readable_quick(path: str) -> bool:
    """Quickly test if a TIFF file is readable by attempting to read a small window."""
    try:
        with rasterio.open(path) as src:
            # Attempt to read a small window of the Tiff file
            # Read all column of the first row
            _ = src.read(1, window=((0, 1), (0, src.width)))
        return True
    except Exception:
        return False


class RadarDataset(Dataset):
    """Dataset for radar nowcasting sequences stored as TIFF images."""

    def __init__(
        self,
        data_path: str,
        input_length: int = 18,
        pred_length: int = 6,
        is_train: bool = True,
        generate_mask: bool = True,
        return_paths: bool = False,
        min_size: int = 1024,
    ):
        self.input_length = input_length
        self.pred_length = pred_length
        self.seq_length = input_length + pred_length
        self.min_size = min_size

        self.files: List[str] = []
        # Add to the 'files' list, recursively, all .tif files
        self.files += glob.glob(os.path.join(data_path, "**/*.tif"), recursive=True)
        # Add to the 'files' list, recursively, all .tiff files
        self.files += glob.glob(os.path.join(data_path, "**/*.tiff"), recursive=True)
        # Sort alphabetically
        self.files = sorted(self.files)

        if len(self.files) == 0:
            raise RuntimeError(f"No files found in {data_path}.")
        if len(self.files) < self.seq_length:
            raise RuntimeError(
                f"Not enough files in {data_path}: {len(self.files)} < seq_length={self.seq_length}"
            )

        self.is_train = is_train
        # Pipeline of transformation to apply to images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # Convert a Numpy array or PIL Image to a tensor --> [ C, H, W ]
            # Normalize to [0, 1]
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.file_validity = {}
        self.valid_indices: List[int] = []
        self.total_possible_windows = max(0, len(self.files) - self.seq_length + 1)
        self.augmentation_transforms = get_augmentation_transforms() if is_train else None

        # Check and consider only the valid sequences of files for the train or test
        # If a sequence contains also a single invalid file , then the sequence doesn't not considered
        for start_idx in range(self.total_possible_windows):
            ok = True
            for i in range(self.seq_length):
                f = self.files[start_idx + i]
                if f not in self.file_validity:
                    try:
                        size = os.path.getsize(f)
                    except FileNotFoundError:
                        size = 0
                    if size < self.min_size:
                        valid = False
                    else:
                        valid = tiff_is_readable_quick(f)
                        if not valid:
                            print(f"Invalid file (read fail): {f}")
                    self.file_validity[f] = valid
                if not self.file_validity[f]:
                    ok = False
                    break
            if ok:
                self.valid_indices.append(start_idx)

        self.total_files = len(self.files)
        self.invalid_files = sum(1 for v in self.file_validity.values() if not v)
        self.valid_windows = len(self.valid_indices)
        self.invalid_windows = self.total_possible_windows - self.valid_windows

        print(f"\nDataset Statistics ({'train' if is_train else 'val'}):")
        print(f"1. Total files: {self.total_files}")
        print(f"2. Invalid files: {self.invalid_files}")
        print(f"3. Total possible windows: {self.total_possible_windows}")
        print(f"4. Valid windows: {self.valid_windows}")
        print(f"5. Invalid windows: {self.invalid_windows}")
        print(" ===================================================== \n")

        self.generate_mask = generate_mask
        self.return_paths = return_paths

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        max_resamples = 8
        attempts = 0
        base_idx = idx

        while attempts < max_resamples:
            start = self.valid_indices[idx]
            images = []
            target_frame_paths: List[str] = []
            failed = False

            for i in range(self.seq_length):
                f = self.files[start + i]
                try:
                    with rasterio.open(f) as src:
                        # Reads the first band of the raster file  
                        # Converts the read data to float32 type
                        img = src.read(1).astype(np.float32)
                except Exception:
                    # If the file is invalid
                    self.file_validity[f] = False
                    failed = True
                    if self.is_train:
                        # Change randomically the position of idx with a casual jump between 1 and 63
                        idx = (idx + np.random.randint(1, 64)) % len(self.valid_indices)
                    else:
                        idx = (idx + 1) % len(self.valid_indices)
                    attempts += 1
                    break

                img = normalize_image(img)
                img = Image.fromarray((img * 255.0).astype(np.uint8))
                # Applys the piple of trasformation
                img = self.transform(img)
                images.append(img)

                if i >= self.input_length:
                    target_frame_paths.append(f)

            if failed:
                continue

            # 'images' --> list of tensor [ C, H, W ]
            # 'all_frames' --> tensor [ N (num. of images), C, H, W ]
            all_frames = torch.stack(images)
            # Permute the tensor from --> [ N, C, H, W] to [ C, N, H, W ]
            all_frames = all_frames.permute(1, 0, 2, 3)

            if self.is_train and self.augmentation_transforms is not None:
                all_frames = self.augmentation_transforms(all_frames)

            if not isinstance(all_frames, torch.Tensor):
                all_frames = torch.as_tensor(all_frames)

            # Slicing operation --> extract only N=input_length image from the sequence
            inputs = all_frames[:, :self.input_length]
            # Slicing operations --> extract only N=input_length image from the seqence
            targets = all_frames[:, self.input_length:]

             # Permute the tensor from --> [ N, C, H, W] to [ C, N, H, W ]
            inputs = inputs.permute(1, 0, 2, 3)
             # Permute the tensor from --> [ N, C, H, W] to [ C, N, H, W ]
            targets = targets.permute(1, 0, 2, 3)


            # From target images , masks the value lower than -1.0 with 0.0 , 1.0 otherwise
            mask: Optional[torch.Tensor] = None
            if self.generate_mask:
                mask = torch.where(targets > -1.0, 1.0, 0.0)

            if self.return_paths:
                if len(target_frame_paths) != self.pred_length:
                    raise RuntimeError(
                        f"target_frame_paths len={len(target_frame_paths)} differs from "
                        f"pred_length={self.pred_length} for window start index {start}."
                    )
                return inputs, targets, mask, target_frame_paths

            return inputs, targets, mask

        raise RasterioIOError(
            f"No valid window found after {max_resamples} attempts; original index {base_idx}."
        )


def collate_val(batch):
    """Custom collate function for validation loader with batch_size=1."""
    item = batch[0]
    if len(item) != 4:
        raise RuntimeError("Validation loader must return (inputs, targets, mask, paths).")
    inputs, targets, mask, paths = item

    if isinstance(paths, (list, tuple)) and paths and isinstance(paths[0], (list, tuple)):
        paths = list(paths[0])
    elif isinstance(paths, (list, tuple)):
        paths = list(paths)
    else:
        paths = [paths]

    if inputs.dim() == 4:
        inputs = inputs.unsqueeze(0)
    if targets.dim() == 4:
        targets = targets.unsqueeze(0)
    if mask is not None and mask.dim() == 4:
        mask = mask.unsqueeze(0)

    return inputs, targets, mask, paths


def create_dataloaders(
    data_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    pred_length: int = PRED_LENGTH,
    small_debug: bool = False,
    debug_train_size: int = 64,
    debug_val_size: int = 16,
):
    """Create training and validation DataLoaders for the radar dataset."""
    set_seed(15)

    train_dataset = RadarDataset(
        os.path.join(data_path, "train"),
        is_train=True,
        pred_length=pred_length,
    )
    val_dataset = RadarDataset(
        os.path.join(data_path, "val"),
        is_train=False,
        return_paths=True,
        pred_length=pred_length,
    )

    if small_debug:
        train_len = min(debug_train_size, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(train_len)))
        val_len = min(debug_val_size, len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(val_len)))
        print(f"[small-debug] Using {train_len} train samples and {val_len} val samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_val,
    )

    return train_loader, val_loader
