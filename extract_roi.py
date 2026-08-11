import os
import cv2
import numpy as np

from tqdm import tqdm
from tensorflow.keras.models import load_model


MODEL_PATH = "models/final_unet.keras"

INPUT_DIR = "processed"

OUTPUT_DIR = "segmented_roi"

IMG_SIZE = (256, 256)

THRESHOLD = 0.5

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print(
    "Loading trained U-Net model..."
)

model = load_model(
    MODEL_PATH,
    compile=False
)


def clean_mask(mask):

    mask = (
        mask > THRESHOLD
    ).astype(
        np.uint8
    ) * 255

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def extract_region(
    original_img,
    mask_256
):

    h, w = (
        original_img.shape[:2]
    )

    mask = cv2.resize(
        mask_256,
        (w, h)
    )

    mask = clean_mask(
        mask
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        return original_img

    largest = max(
        contours,
        key=cv2.contourArea
    )

    x, y, bw, bh = cv2.boundingRect(
        largest
    )

    roi = original_img[
        y:y + bh,
        x:x + bw
    ]

    roi_mask = mask[
        y:y + bh,
        x:x + bw
    ]

    roi = cv2.bitwise_and(
        roi,
        roi,
        mask=roi_mask
    )

    return roi


def process_folder():

    classes = [
        d for d in os.listdir(
            INPUT_DIR
        )
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
            for f in os.listdir(
                in_dir
            )
            if f.lower().endswith(
                (".jpg", ".jpeg")
            )
        ]

        for fname in tqdm(
            files,
            desc=f"Extracting ROI for {cls}"
        ):

            path = os.path.join(
                in_dir,
                fname
            )

            img = cv2.imread(
                path
            )

            if img is None:
                continue

            img_resized = cv2.resize(
                img,
                IMG_SIZE
            )

            inp = (
                img_resized.astype(
                    "float32"
                ) / 255.0
            )

            inp = np.expand_dims(
                inp,
                axis=0
            )

            pred_mask = model.predict(
                inp,
                verbose=0
            )[0, :, :, 0]

            roi = extract_region(
                img,
                pred_mask
            )

            cv2.imwrite(
                os.path.join(
                    out_dir,
                    fname
                ),
                roi,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )


if __name__ == "__main__":

    process_folder()

    print(
        "ROI extraction complete."
    )