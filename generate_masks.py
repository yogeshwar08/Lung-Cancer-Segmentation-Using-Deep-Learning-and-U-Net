import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "processed"
OUTPUT_DIR = "masks_auto"

IMG_SIZE = (256, 256)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def create_mask(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    if np.mean(gray[mask == 255]) > np.mean(
        gray[mask == 0]
    ):
        mask = cv2.bitwise_not(mask)

    mask = cv2.medianBlur(
        mask,
        5
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=3
    )

    return mask


def generate_all():

    classes = [
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(
            os.path.join(
                INPUT_DIR,
                d
            )
        )
    ]

    for cls in classes:

        in_dir = os.path.join(
            INPUT_DIR,
            cls
        )

        out_dir = os.path.join(
            OUTPUT_DIR,
            cls
        )

        os.makedirs(
            out_dir,
            exist_ok=True
        )

        files = [
            f
            for f in os.listdir(in_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg")
            )
        ]

        for fname in tqdm(
            files,
            desc=f"Masks for {cls}"
        ):

            img_path = os.path.join(
                in_dir,
                fname
            )

            img = cv2.imread(
                img_path
            )

            if img is None:
                continue

            img = cv2.resize(
                img,
                IMG_SIZE
            )

            mask = create_mask(
                img
            )

            save_path = os.path.join(
                out_dir,
                fname
            )

            cv2.imwrite(
                save_path,
                mask
            )


if __name__ == "__main__":

    generate_all()

    print(
        "Auto mask generation complete."
    )