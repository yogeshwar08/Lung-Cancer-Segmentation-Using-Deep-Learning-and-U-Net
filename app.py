import os
import uuid

import cv2
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# SETTINGS
# ============================================================

UPLOAD_FOLDER = "static/uploads"

CLASSIFIER_MODEL_PATH = "models/RS_CapsNet.pth"
CLASS_PATH = "models/classes.txt"
UNET_MODEL_PATH = "models/final_unet.keras"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

# Render deployment uses CPU
device = torch.device("cpu")
torch.set_num_threads(1)

print("Using device:", device)


# ============================================================
# RS-CAPSNET CLASSIFIER
# ============================================================

class RS_CapsNet(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)
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

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(
        "models/classes.txt not found."
    )

with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]

if not class_names:
    raise ValueError(
        "No class names found in models/classes.txt"
    )

print("Classes:", class_names)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

if not os.path.exists(CLASSIFIER_MODEL_PATH):
    raise FileNotFoundError(
        "models/RS_CapsNet.pth not found."
    )

classifier = RS_CapsNet(
    num_classes=len(class_names)
)

classifier.load_state_dict(
    torch.load(
        CLASSIFIER_MODEL_PATH,
        map_location="cpu"
    )
)

classifier = classifier.to(device)
classifier.eval()

print("RS-CapsNet loaded successfully.")


# ============================================================
# LOAD U-NET SEGMENTATION MODEL
# ============================================================

if not os.path.exists(UNET_MODEL_PATH):
    raise FileNotFoundError(
        "models/final_unet.keras not found. "
        "The prediction gate requires the U-Net model."
    )

unet_model = tf.keras.models.load_model(
    UNET_MODEL_PATH,
    compile=False
)

print("U-Net loaded successfully.")


# ============================================================
# CLASSIFIER TRANSFORMATION
# ============================================================

classifier_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ============================================================
# CT INPUT VALIDATION
# ============================================================

def looks_like_ct_image(image_bgr):
    """
    Reject obvious non-CT photographs before running the
    classifier.

    CT images are normally grayscale. A normal family photo,
    selfie, outdoor photo, etc. usually has substantially
    different RGB channels and/or high color saturation.
    """

    if image_bgr is None:
        return False, "Invalid image."

    h, w = image_bgr.shape[:2]

    if h < 64 or w < 64:
        return False, "Image resolution is too small."

    # Check how different the color channels are.
    b, g, r = cv2.split(image_bgr)

    channel_difference = (
        np.mean(np.abs(b.astype(np.float32) - g.astype(np.float32))) +
        np.mean(np.abs(g.astype(np.float32) - r.astype(np.float32))) +
        np.mean(np.abs(b.astype(np.float32) - r.astype(np.float32)))
    ) / 3.0

    # Mean HSV saturation: CT scans should normally have very
    # little actual color information.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    if channel_difference > 12.0 or mean_saturation > 35.0:
        return False, (
            "This does not look like a grayscale lung CT image. "
            "Please upload a valid lung CT JPEG image."
        )

    return True, ""


# ============================================================
# U-NET MASK + ROI VALIDATION
# ============================================================

def clean_mask(mask):
    mask = (mask > 0.5).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)

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


def get_lung_roi(image_bgr):
    """
    Run the same U-Net -> mask -> largest contour -> ROI concept
    used by the training pipeline.

    Returns:
        roi, diagnostics
    """

    h, w = image_bgr.shape[:2]

    # U-Net training uses 256x256 RGB images normalized to [0, 1].
    img_256 = cv2.resize(
        image_bgr,
        (256, 256)
    )

    inp = img_256.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)

    pred_mask = unet_model.predict(
        inp,
        verbose=0
    )[0, :, :, 0]

    mask = clean_mask(pred_mask)

    # Convert mask back to original image size.
    mask = cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, (
            "No lung-like region was detected. "
            "Please upload a lung CT image."
        )

    largest = max(
        contours,
        key=cv2.contourArea
    )

    contour_area = float(cv2.contourArea(largest))
    image_area = float(h * w)
    contour_ratio = contour_area / image_area

    x, y, bw, bh = cv2.boundingRect(largest)

    # Basic segmentation sanity checks.
    # These are deliberately conservative enough to avoid rejecting
    # ordinary CT scans while blocking obvious unrelated images.
    if contour_ratio < 0.01:
        return None, (
            "A valid lung region could not be detected. "
            "Please upload a lung CT image."
        )

    if contour_ratio > 0.75:
        return None, (
            "The detected region is not consistent with a lung CT scan. "
            "Please upload a valid lung CT image."
        )

    if bw < 32 or bh < 32:
        return None, (
            "The detected lung region is too small. "
            "Please upload a clear lung CT image."
        )

    aspect_ratio = bw / float(bh)

    if aspect_ratio < 0.20 or aspect_ratio > 5.0:
        return None, (
            "The detected region does not look like a lung CT region. "
            "Please upload a valid lung CT image."
        )

    # Extract ROI exactly in the spirit of extract_roi.py.
    roi = image_bgr[
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

    if roi.size == 0:
        return None, (
            "Unable to extract a valid lung region."
        )

    diagnostics = {
        "mask_ratio": contour_ratio,
        "bbox_width": bw,
        "bbox_height": bh,
        "roi": roi
    }

    return roi, diagnostics


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("predict.html")

    filepath = None

    try:

        # --------------------------------------------------------
        # CHECK UPLOAD
        # --------------------------------------------------------

        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded."
            }), 400

        file = request.files["image"]

        if not file.filename:
            return jsonify({
                "error": "No file selected."
            }), 400

        original_filename = file.filename.lower()

        if not original_filename.endswith(
            (".jpg", ".jpeg")
        ):
            return jsonify({
                "error":
                "Only JPG and JPEG images are allowed."
            }), 400

        # --------------------------------------------------------
        # SAVE TEMP IMAGE
        # --------------------------------------------------------

        filename = (
            uuid.uuid4().hex
            + "_"
            + os.path.basename(file.filename)
        )

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        # --------------------------------------------------------
        # READ ORIGINAL IMAGE
        # --------------------------------------------------------

        image_bgr = cv2.imread(filepath)

        if image_bgr is None:
            return jsonify({
                "error": "Could not read the uploaded image."
            }), 400

        # --------------------------------------------------------
        # STEP 1: REJECT OBVIOUS NON-CT IMAGES
        # --------------------------------------------------------

        valid_input, input_message = looks_like_ct_image(
            image_bgr
        )

        if not valid_input:
            return render_template(
                "result.html",
                predicted_class="Invalid Input",
                confidence=0,
                probs=[],
                error_message=input_message
            )

        # --------------------------------------------------------
        # STEP 2: U-NET VALIDATION / ROI EXTRACTION
        # --------------------------------------------------------

        roi, roi_result = get_lung_roi(
            image_bgr
        )

        if roi is None:
            return render_template(
                "result.html",
                predicted_class="Invalid Input",
                confidence=0,
                probs=[],
                error_message=roi_result
            )

        # --------------------------------------------------------
        # STEP 3: CLASSIFY ONLY THE VALIDATED ROI
        # --------------------------------------------------------

        roi_rgb = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            roi_rgb
        )

        input_tensor = classifier_transform(
            image
        )

        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.inference_mode():

            output = classifier(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

        predicted_index = int(
            probabilities.argmax().item()
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = (
            float(
                probabilities[
                    predicted_index
                ].item()
            ) * 100
        )

        # --------------------------------------------------------
        # STEP 4: CONFIDENCE GATE
        # --------------------------------------------------------
        # Softmax can be overconfident. Therefore a low-confidence
        # result is not forced into benign/malignant/normal.

        MIN_CONFIDENCE = 60.0

        if confidence < MIN_CONFIDENCE:
            predicted_class = "Uncertain"
            display_confidence = confidence
        else:
            display_confidence = confidence

        # --------------------------------------------------------
        # ALL PROBABILITIES
        # --------------------------------------------------------

        probs = []

        for i in range(len(class_names)):

            probs.append({
                "name": class_names[i],
                "prob": round(
                    float(
                        probabilities[i].item()
                    ),
                    4
                )
            })

        print(
            "Prediction:",
            predicted_class
        )

        print(
            "Confidence:",
            confidence
        )

        # --------------------------------------------------------
        # RESULT PAGE
        # --------------------------------------------------------

        return render_template(
            "result.html",
            predicted_class=predicted_class,
            confidence=display_confidence,
            probs=probs,
            error_message=None
        )

    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return jsonify({
            "error": "Prediction failed.",
            "details": str(e)
        }), 500

    finally:

        # --------------------------------------------------------
        # DELETE TEMPORARY IMAGE
        # --------------------------------------------------------

        if filepath and os.path.exists(filepath):

            try:
                os.remove(filepath)

                print(
                    "Temporary image removed."
                )

            except Exception as cleanup_error:

                print(
                    "Could not remove temporary image:",
                    cleanup_error
                )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "model": "loaded",
        "unet": "loaded",
        "device": str(device),
        "classes": class_names
    })


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )