import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import RoadDataset
from src.models.unet import UNet


def dice_loss(logits, targets):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))

    score = (
        (2 * intersection + 1.0)
        / (
            probabilities.sum(dim=(1, 2, 3))
            + targets.sum(dim=(1, 2, 3))
            + 1.0
        )
    )

    return 1 - score.mean()


def main():
    torch.set_num_threads(4)

    manifest = r"data\processed\manifest.csv"

    train_dataset = RoadDataset(manifest, split="train", size=512)
    validation_dataset = RoadDataset(manifest, split="val", size=256)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cpu")
    model = UNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([20.0])
    )

    best_validation_loss = float("inf")

    for epoch in range(1, 21):
        model.train()
        training_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            predictions = model(images)

            loss = (
                0.5 * bce(predictions, masks)
                + 0.5 * dice_loss(predictions, masks)
            )

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for images, masks in validation_loader:
                images = images.to(device)
                masks = masks.to(device)

                predictions = model(images)

                loss = (
                    0.5 * bce(predictions, masks)
                    + 0.5 * dice_loss(predictions, masks)
                )

                validation_loss += loss.item()

        training_loss /= len(train_loader)
        validation_loss /= len(validation_loader)

        print(
            f"Epoch {epoch}/5 | "
            f"Train Loss: {training_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                model.state_dict(),
                r"outputs\unet_best.pt"
            )
            print("Best model saved")


if __name__ == "__main__":
    main()