import csv
import random

import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset


def add_occlusion(image):
    if random.random() > 0.8:
        return image

    _, height, width = image.shape
    output = image.clone()

    yy, xx = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij"
    )

    for _ in range(random.randint(1, 3)):
        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)
        radius_x = random.randint(25, 80)
        radius_y = random.randint(25, 80)

        blob = (
            ((xx - center_x) / radius_x) ** 2
            + ((yy - center_y) / radius_y) ** 2
            <= 1
        )

        if random.random() < 0.5:
            color = torch.rand(3) * 0.15
        else:
            color = torch.ones(3) * random.uniform(0.75, 1.0)

        output[:, blob] = color[:, None]

    return output.clamp(0, 1)


class RoadDataset(Dataset):
    def __init__(self, manifest_path, split="train", size=512):
        with open(manifest_path, newline="") as file:
            all_rows = list(csv.DictReader(file))

        self.rows = [
            row for row in all_rows
            if row["split"] == split
        ]

        self.split = split
        self.size = size
        self.patches_per_image = 2 if split == "train" else 1

    def __len__(self):
        return len(self.rows) * self.patches_per_image

    def __getitem__(self, index):
        row = self.rows[index % len(self.rows)]

        with rasterio.open(row["image_path"]) as source:
            image = source.read([1, 2, 3]).astype(np.float32)

        image = np.moveaxis(image, 0, -1)

        low = np.percentile(
            image, 2, axis=(0, 1), keepdims=True
        )
        high = np.percentile(
            image, 98, axis=(0, 1), keepdims=True
        )

        image = np.clip(
            (image - low) / (high - low + 1e-6),
            0,
            1
        )

        mask = np.asarray(
            Image.open(row["mask_path"]).convert("L")
        )

        height, width = mask.shape

        if (
            self.split == "train"
            and height >= self.size
            and width >= self.size
        ):
            for attempt in range(20):
                top = random.randint(0, height - self.size)
                left = random.randint(0, width - self.size)

                image_crop = image[
                    top:top + self.size,
                    left:left + self.size
                ]

                mask_crop = mask[
                    top:top + self.size,
                    left:left + self.size
                ]

                if mask_crop.mean() > 1.0 or attempt == 19:
                    image = image_crop
                    mask = mask_crop
                    break

        image = Image.fromarray(
            (image * 255).astype(np.uint8)
        ).resize(
            (self.size, self.size),
            Image.Resampling.BILINEAR
        )

        mask = Image.fromarray(mask).resize(
            (self.size, self.size),
            Image.Resampling.NEAREST
        )

        image = torch.from_numpy(
            np.asarray(image).copy()
        ).permute(2, 0, 1).float() / 255.0

        mask = torch.from_numpy(
            (np.asarray(mask).copy() > 0).astype(np.float32)
        ).unsqueeze(0)

        if self.split == "train":
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])

            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])

            image = add_occlusion(image)

        return image, mask