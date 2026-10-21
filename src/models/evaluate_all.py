from pathlib import Path
import csv

import pandas as pd
import torch

from src.models.unet import UNet
from src.models.evaluate_tiled import (
    normalize_image,
    predict_tiled,
    load_mask,
    calculate_metrics
)


MANIFEST_PATH = r"data\processed\manifest.csv"
MODEL_PATH = r"outputs\unet_best.pt"
OUTPUT_PATH = r"outputs\test_metrics.csv"

THRESHOLD = 0.35


def main():
    with open(MANIFEST_PATH, newline="") as file:
        rows = list(csv.DictReader(file))

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    model = UNet()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    results = []

    for index, row in enumerate(test_rows, start=1):
        image_path = row["image_path"]

        print(
            f"Evaluating test image "
            f"{index}/{len(test_rows)}"
        )

        image = normalize_image(image_path)
        probability = predict_tiled(model, image)
        mask = load_mask(row["mask_path"])

        iou, dice = calculate_metrics(
            probability,
            mask,
            THRESHOLD
        )

        image_name = Path(image_path).stem

        results.append(
            {
                "image": image_name,
                "iou": iou,
                "dice": dice
            }
        )

    results_table = pd.DataFrame(results)

    results_table.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(results_table)
    print()
    print(
        "Average IoU:",
        results_table["iou"].mean()
    )

    print(
        "Average Dice:",
        results_table["dice"].mean()
    )

    print(
        "Saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()