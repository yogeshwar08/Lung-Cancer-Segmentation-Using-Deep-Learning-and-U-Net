import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "dataset"
OUTPUT_DIR = "processed"

# JPEG ONLY
IMG_EXT = ("*.jpg", "*.jpeg")

RGF_ITER = 3
BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

MEDIAN_KSIZE = 3
VAR_THRESHOLD = 500.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


def rolling_guidance_filter(
    img,
    iterations=3,
    d=9,
    sigmaColor=75,
    sigmaSpace=75
):
    img_f = img.copy()

    for _ in range(iterations):
        img_f = cv2.bilateralFilter(
            img_f,
            d,
            sigmaColor,
            sigmaSpace
        )

    return img_f


def switched_median_filter(
    img_gray,
    ksize=3,
    var_threshold=500.0
):
    padded = cv2.copyMakeBorder(
        img_gray,
        ksize // 2,
        ksize // 2,
        ksize // 2,
        ksize // 2,
        cv2.BORDER_REFLECT
    )

    out = img_gray.copy()

    h, w = img_gray.shape

    for i in range(h):
        for j in range(w):

            win = padded[
                i:i + ksize,
                j:j + ksize
            ]

            if win.var() > var_threshold:
                out[i, j] = np.median(win)

    return out


def preprocess_image(path, save_path):

    img = cv2.imread(path)

    if img is None:
        return False

    rgf = rolling_guidance_filter(
        img,
        RGF_ITER,
        BILATERAL_DIAMETER,
        BILATERAL_SIGMA_COLOR,
        BILATERAL_SIGMA_SPACE
    )

    gray = cv2.cvtColor(
        rgf,
        cv2.COLOR_BGR2GRAY
    )

    smf = switched_median_filter(
        gray,
        MEDIAN_KSIZE,
        VAR_THRESHOLD
    )

    ycrcb = cv2.cvtColor(
        rgf,
        cv2.COLOR_BGR2YCrCb
    )

    ycrcb[:, :, 0] = smf

    fused = cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2BGR
    )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    cv2.imwrite(
        save_path,
        fused,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    return True


def process_dataset():

    classes = [
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(
            os.path.join(INPUT_DIR, d)
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

        files = []

        for ext in IMG_EXT:
            files.extend(
                glob(
                    os.path.join(
                        in_dir,
                        ext
                    )
                )
            )

        for path in tqdm(
            files,
            desc=f"Processing {cls}"
        ):

            filename = os.path.basename(path)

            save_path = os.path.join(
                out_dir,
                filename
            )

            preprocess_image(
                path,
                save_path
            )


if __name__ == "__main__":

    process_dataset()

    print(
        "Preprocessing complete."
    )
    