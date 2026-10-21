import csv
import numpy as np
import rasterio
import torch

from PIL import Image
from skimage.morphology import dilation, disk

from src.models.unet import UNet


MANIFEST_PATH = r"data\processed\manifest.csv"
MODEL_PATH = r"outputs\unet_best.pt"

TILE_SIZE = 512
STRIDE = 384


def normalize_image(path):
    with rasterio.open(path) as source:
        image = source.read([1, 2, 3]).astype(np.float32)

    image = np.moveaxis(image, 0, -1)

    low = np.percentile(
        image, 2, axis=(0, 1), keepdims=True
    )
    high = np.percentile(
        image, 98, axis=(0, 1), keepdims=True
    )

    image = (image - low) / (high - low + 1e-6)
    return np.clip(image, 0, 1)


def get_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last_position = length - tile_size

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


def predict_tiled(model, image):
    height, width, _ = image.shape

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    rows = get_positions(
        height,
        TILE_SIZE,
        STRIDE
    )

    columns = get_positions(
        width,
        TILE_SIZE,
        STRIDE
    )

    with torch.no_grad():
        for top in rows:
            for left in columns:
                tile = image[
                    top:top + TILE_SIZE,
                    left:left + TILE_SIZE
                ]

                tensor = torch.from_numpy(
                    tile.copy()
                ).permute(2, 0, 1).float().unsqueeze(0)

                output = model(tensor)
                prediction = torch.sigmoid(output)

                prediction = prediction[
                    0, 0
                ].cpu().numpy()

                probability_sum[
                    top:top + TILE_SIZE,
                    left:left + TILE_SIZE
                ] += prediction

                count[
                    top:top + TILE_SIZE,
                    left:left + TILE_SIZE
                ] += 1

    return probability_sum / np.maximum(count, 1)


def load_rows(split):
    with open(MANIFEST_PATH, newline="") as file:
        rows = list(csv.DictReader(file))

    return [
        row for row in rows
        if row["split"] == split
    ]


def load_mask(path):
    mask = np.asarray(
        Image.open(path).convert("L")
    ) > 0

    return dilation(mask, disk(3))


def calculate_metrics(probability, mask, threshold):
    prediction = probability > threshold
    truth = mask > 0

    intersection = np.logical_and(
        prediction,
        truth
    ).sum()

    union = np.logical_or(
        prediction,
        truth
    ).sum()

    iou = intersection / (union + 1e-6)

    dice = (
        2 * intersection
        / (prediction.sum() + truth.sum() + 1e-6)
    )

    return float(iou), float(dice)


def evaluate_items(items, threshold):
    scores = []

    for probability, mask in items:
        scores.append(
            calculate_metrics(
                probability,
                mask,
                threshold
            )
        )

    mean_iou = np.mean(
        [score[0] for score in scores]
    )

    mean_dice = np.mean(
        [score[1] for score in scores]
    )

    return float(mean_iou), float(mean_dice)


def main():
    device = torch.device("cpu")

    model = UNet().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    validation_rows = load_rows("val")
    test_rows = load_rows("test")

    validation_items = []
    test_items = []

    print("Running tiled validation inference...")

    for index, row in enumerate(validation_rows, start=1):
        image = normalize_image(row["image_path"])
        probability = predict_tiled(model, image)
        mask = load_mask(row["mask_path"])

        validation_items.append(
            (probability, mask)
        )

        print(
            f"Validation image {index}/{len(validation_rows)}"
        )

    print("Running tiled test inference...")

    for index, row in enumerate(test_rows, start=1):
        image = normalize_image(row["image_path"])
        probability = predict_tiled(model, image)
        mask = load_mask(row["mask_path"])

        test_items.append(
            (probability, mask)
        )

        print(
            f"Test image {index}/{len(test_rows)}"
        )

    thresholds = [
        round(value / 100, 2)
        for value in range(30, 81, 5)
    ]

    best_threshold = 0.5
    best_validation_iou = -1
    best_validation_dice = 0

    for threshold in thresholds:
        validation_iou, validation_dice = evaluate_items(
            validation_items,
            threshold
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"Validation IoU: {validation_iou:.4f} | "
            f"Validation Dice: {validation_dice:.4f}"
        )

        if validation_iou > best_validation_iou:
            best_validation_iou = validation_iou
            best_validation_dice = validation_dice
            best_threshold = threshold

    test_iou, test_dice = evaluate_items(
        test_items,
        best_threshold
    )

    print()
    print(f"Best threshold: {best_threshold}")
    print(f"Validation IoU: {best_validation_iou}")
    print(f"Validation Dice: {best_validation_dice}")
    print(f"Test IoU: {test_iou}")
    print(f"Test Dice: {test_dice}")


if __name__ == "__main__":
    main()