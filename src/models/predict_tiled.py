import csv
import numpy as np
import rasterio
import torch

from PIL import Image

from src.models.unet import UNet


MANIFEST_PATH = r"data\processed\manifest.csv"
MODEL_PATH = r"outputs\unet_best.pt"

OUTPUT_MASK = r"outputs\predicted_tiled_mask.png"
OUTPUT_OVERLAY = r"outputs\predicted_tiled_overlay.png"

TILE_SIZE = 512
STRIDE = 384
THRESHOLD = 0.50


def normalize_image(path):
    with rasterio.open(path) as source:
        image = source.read([1, 2, 3]).astype(np.float32)

    image = np.moveaxis(image, 0, -1)

    low = np.percentile(
        image,
        2,
        axis=(0, 1),
        keepdims=True
    )

    high = np.percentile(
        image,
        98,
        axis=(0, 1),
        keepdims=True
    )

    image = (image - low) / (high - low + 1e-6)

    return np.clip(image, 0, 1)


def get_positions(length):
    if length <= TILE_SIZE:
        return [0]

    positions = list(
        range(
            0,
            length - TILE_SIZE + 1,
            STRIDE
        )
    )

    last_position = length - TILE_SIZE

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

    row_positions = get_positions(height)
    column_positions = get_positions(width)

    total_tiles = len(row_positions) * len(column_positions)
    tile_number = 0

    with torch.no_grad():
        for top in row_positions:
            for left in column_positions:
                tile_number += 1

                tile = image[
                    top:top + TILE_SIZE,
                    left:left + TILE_SIZE
                ]

                tensor = torch.from_numpy(
                    tile.copy()
                ).permute(2, 0, 1).float().unsqueeze(0)

                output = model(tensor)

                prediction = torch.sigmoid(
                    output
                )[0, 0].cpu().numpy()

                probability_sum[
                    top:top + TILE_SIZE,
                    left:left + TILE_SIZE
                ] += prediction

                count[
                    top:top + TILE_SIZE,
                    left:left + TILE_SIZE
                ] += 1

                print(
                    f"Processed tile {tile_number}/{total_tiles}"
                )

    return probability_sum / np.maximum(count, 1)


def main():
    with open(MANIFEST_PATH, newline="") as file:
        rows = list(csv.DictReader(file))

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    selected_row = test_rows[0]

    image_path = selected_row["image_path"]

    print("Input image:", image_path)

    image = normalize_image(image_path)

    model = UNet()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    probability = predict_tiled(
        model,
        image
    )

    prediction = probability > THRESHOLD

    mask = (
        prediction.astype(np.uint8) * 255
    )

    Image.fromarray(mask).save(
        OUTPUT_MASK
    )

    base_image = Image.fromarray(
        (image * 255).astype(np.uint8)
    )

    red_image = Image.new(
        "RGB",
        base_image.size,
        (255, 0, 0)
    )

    mask_image = Image.fromarray(mask)

    red_overlay = Image.composite(
        red_image,
        base_image,
        mask_image
    )

    final_overlay = Image.blend(
        base_image,
        red_overlay,
        0.45
    )

    final_overlay.save(
        OUTPUT_OVERLAY
    )

    print("Probability maximum:", float(probability.max()))
    print("Predicted road pixels:", int(prediction.sum()))
    print("Saved:", OUTPUT_MASK)
    print("Saved:", OUTPUT_OVERLAY)


if __name__ == "__main__":
    main()