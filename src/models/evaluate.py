import torch

from src.data.dataset import RoadDataset
from src.models.unet import UNet


def calculate_scores(model, dataset, threshold):
    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for index in range(len(dataset)):
            image, mask = dataset[index]

            output = model(image.unsqueeze(0))
            probability = torch.sigmoid(output)[0, 0]

            prediction = probability > threshold
            truth = mask[0] > 0.5

            intersection = (
                prediction & truth
            ).sum().item()

            union = (
                prediction | truth
            ).sum().item()

            iou = intersection / (union + 1e-6)

            dice = (
                2 * intersection
                / (
                    prediction.sum().item()
                    + truth.sum().item()
                    + 1e-6
                )
            )

            iou_scores.append(iou)
            dice_scores.append(dice)

    mean_iou = sum(iou_scores) / len(iou_scores)
    mean_dice = sum(dice_scores) / len(dice_scores)

    return mean_iou, mean_dice


def main():
    model = UNet()

    model.load_state_dict(
        torch.load(
            r"outputs\unet_best.pt",
            map_location="cpu"
        )
    )

    model.eval()

    manifest = r"data\processed\manifest.csv"

    validation_dataset = RoadDataset(
        manifest,
        split="val",
        size=512
    )

    test_dataset = RoadDataset(
        manifest,
        split="test",
        size=512
    )

    thresholds = [
        round(value / 100, 2)
        for value in range(10, 91, 5)
    ]

    validation_results = []

    for threshold in thresholds:
        iou, dice = calculate_scores(
            model,
            validation_dataset,
            threshold
        )

        validation_results.append(
            (threshold, iou, dice)
        )

    best_threshold, best_iou, best_dice = max(
        validation_results,
        key=lambda result: result[1]
    )

    test_iou, test_dice = calculate_scores(
        model,
        test_dataset,
        best_threshold
    )

    print("Best threshold:", best_threshold)
    print("Validation IoU:", best_iou)
    print("Validation Dice:", best_dice)
    print("Test IoU:", test_iou)
    print("Test Dice:", test_dice)


if __name__ == "__main__":
    main()