import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# ==================================================
# SETTINGS
# ==================================================

DATA_DIR = "segmented_roi"
MODEL_DIR = "models"

EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 0.001
IMAGE_SIZE = 128

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==================================================
# IMAGE TRANSFORMATION
# ==================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.Grayscale(
            num_output_channels=1
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ]
)


# ==================================================
# RS-CAPSNET MODEL
# ==================================================

class RS_CapsNet(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(
            64 * 32 * 32,
            128
        )

        self.fc2 = nn.Linear(
            128,
            num_classes
        )


    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.pool(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x


# ==================================================
# EVALUATION
# ==================================================

def evaluate(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(
                images
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    if total == 0:

        return 0.0

    return correct / total


# ==================================================
# MAIN
# ==================================================

def main():

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    print("Loading dataset...")

    dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=transform
    )

    print(
        "Total images:",
        len(dataset)
    )

    print(
        "Classes:",
        dataset.classes
    )


    if len(dataset) < 2:

        raise RuntimeError(
            "Not enough images in segmented_roi."
        )


    # --------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------

    train_size = int(
        len(dataset) * 0.75
    )

    validation_size = (
        len(dataset) - train_size
    )


    if train_size == 0:

        train_size = 1

        validation_size = (
            len(dataset) - 1
        )


    if validation_size == 0:

        validation_size = 1

        train_size = (
            len(dataset) - 1
        )


    train_dataset, validation_dataset = (
        random_split(
            dataset,
            [
                train_size,
                validation_size
            ],
            generator=torch.Generator()
            .manual_seed(42)
        )
    )


    print(
        "Training images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(validation_dataset)
    )


    # --------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # --------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------

    model = RS_CapsNet(
        num_classes=len(
            dataset.classes
        )
    )

    model = model.to(device)


    # --------------------------------------------------
    # LOSS AND OPTIMIZER
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    best_accuracy = 0.0

    best_model_path = os.path.join(
        MODEL_DIR,
        "RS_CapsNet.pth"
    )


    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0


        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)


            # Clear gradients

            optimizer.zero_grad()


            # Forward pass

            outputs = model(
                images
            )


            # Calculate loss

            loss = criterion(
                outputs,
                labels
            )


            # Backpropagation

            loss.backward()


            # Update weights

            optimizer.step()


            total_loss += (
                loss.item()
            )


        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        validation_accuracy = evaluate(
            model,
            validation_loader
        )


        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {total_loss:.4f} "
            f"- Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )


        # --------------------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------------------

        if validation_accuracy >= best_accuracy:

            best_accuracy = (
                validation_accuracy
            )

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                "Best model saved."
            )


    # --------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------

    # If the validation accuracy never improved,
    # save the final trained model anyway.

    if not os.path.exists(
        best_model_path
    ):

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "Final model saved."
        )


    # --------------------------------------------------
    # SAVE CLASS NAMES
    # --------------------------------------------------

    class_file = os.path.join(
        MODEL_DIR,
        "classes.txt"
    )


    with open(
        class_file,
        "w",
        encoding="utf-8"
    ) as file:

        for class_name in dataset.classes:

            file.write(
                class_name + "\n"
            )


    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    print()
    print(
        "===================================="
    )

    print(
        "Training complete!"
    )

    print(
        "Best validation accuracy:",
        best_accuracy
    )

    print(
        "Model saved to:",
        best_model_path
    )

    print(
        "Classes saved to:",
        class_file
    )

    print(
        "===================================="
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()