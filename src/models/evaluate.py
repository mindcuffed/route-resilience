import torch

from src.data.dataset import RoadDataset
from src.models.unet import UNet


def main():
    model = UNet()
    model.load_state_dict(
        torch.load(
            r"outputs\unet_best.pt",
            map_location="cpu"
        )
    )
    model.eval()

    dataset = RoadDataset(
        r"data\processed\manifest.csv",
        split="test",
        size=512
    )

    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for index in range(len(dataset)):
            image, mask = dataset[index]

            output = model(image.unsqueeze(0))
            prediction = torch.sigmoid(output)[0, 0] > 0.60
            truth = mask[0] > 0.5

            intersection = (prediction & truth).sum().item()
            union = (prediction | truth).sum().item()

            iou = intersection / (union + 1e-6)
            dice = (
                2 * intersection
                / (prediction.sum().item() + truth.sum().item() + 1e-6)
            )

            iou_scores.append(iou)
            dice_scores.append(dice)

    print("Test images:", len(dataset))
    print("Mean IoU:", sum(iou_scores) / len(iou_scores))
    print("Mean Dice:", sum(dice_scores) / len(dice_scores))


if __name__ == "__main__":
    main()