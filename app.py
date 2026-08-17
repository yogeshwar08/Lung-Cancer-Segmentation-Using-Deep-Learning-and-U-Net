import gc
import io
import logging
import os
from pathlib import Path

# Low-memory CPU configuration must be set before TensorFlow import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

UNET_PATH = MODEL_DIR / "final_unet.keras"
CLASSIFIER_PATH = MODEL_DIR / "RS_CapsNet.pth"
CLASSES_PATH = MODEL_DIR / "classes.txt"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

IMAGE_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Force TensorFlow to CPU and keep inference thread usage low.
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("lungvision")


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

_unet_model = None
_classifier_model = None
_class_names = None


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim, kernel_size=3):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

    def forward(self, x):
        x = self.conv(x)

        batch_size, _, height, width = x.shape

        x = x.view(
            batch_size,
            self.num_capsules,
            self.capsule_dim,
            height,
            width,
        )

        x = x.permute(0, 1, 3, 4, 2)

        x = x.contiguous().view(
            batch_size,
            self.num_capsules * height * width,
            self.capsule_dim,
        )

        return self.squash(x)

    @staticmethod
    def squash(x):
        norm_squared = (x ** 2).sum(dim=-1, keepdim=True)

        scale = norm_squared / (1 + norm_squared)

        return scale * x / torch.sqrt(norm_squared + 1e-8)


class CapsuleLayer(nn.Module):
    def __init__(
        self,
        input_capsules,
        input_dim,
        output_capsules,
        output_dim,
        routing_iterations=3,
    ):
        super().__init__()

        self.input_capsules = input_capsules
        self.input_dim = input_dim
        self.output_capsules = output_capsules
        self.output_dim = output_dim
        self.routing_iterations = routing_iterations

        self.W = nn.Parameter(
            torch.randn(
                1,
                input_capsules,
                output_capsules,
                output_dim,
                input_dim,
            ) * 0.01
        )

    def forward(self, x):
        batch_size = x.size(0)

        W = self.W.expand(
            batch_size,
            -1,
            -1,
            -1,
            -1,
        )

        x = x.unsqueeze(2).unsqueeze(-1)

        u_hat = torch.matmul(W, x).squeeze(-1)

        routing_logits = torch.zeros(
            batch_size,
            self.input_capsules,
            self.output_capsules,
            device=x.device,
        )

        for iteration in range(self.routing_iterations):

            routing_coefficients = torch.softmax(
                routing_logits,
                dim=-1,
            )

            s = (
                routing_coefficients.unsqueeze(-1)
                * u_hat
            ).sum(dim=1)

            v = self.squash(s)

            if iteration < self.routing_iterations - 1:
                agreement = (
                    u_hat * v.unsqueeze(1)
                ).sum(dim=-1)

                routing_logits = routing_logits + agreement

        return v

    @staticmethod
    def squash(x):
        norm_squared = (x ** 2).sum(dim=-1, keepdim=True)

        scale = norm_squared / (1 + norm_squared)

        return scale * x / torch.sqrt(norm_squared + 1e-8)


class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),

            nn.BatchNorm2d(32),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.BatchNorm2d(64),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.BatchNorm2d(128),
        )

        self.primary_caps = PrimaryCaps(
            in_channels=128,
            num_capsules=8,
            capsule_dim=16,
            kernel_size=3,
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )

        self.output_layer = nn.Linear(
            16,
            num_classes,
        )

    def forward(self, x):

        x = self.features(x)

        x = self.primary_caps(x)

        x = x.mean(dim=1)

        x = self.classifier(
            x.unsqueeze(-1)
        ).squeeze(-1)

        return self.output_layer(x)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


def validate_input_image(image):
    """
    Basic validation for uploaded CT-style images.

    This is an input-quality gate, not a medical diagnosis.
    """

    try:
        width, height = image.size

        if width < 128 or height < 128:
            return False, (
                "Please enter a correct lung CT scan image. "
                "The image resolution is too small."
            )

        rgb_image = image.convert("RGB")
        image_array = np.asarray(
            rgb_image,
            dtype=np.float32,
        )

        red = image_array[:, :, 0]
        green = image_array[:, :, 1]
        blue = image_array[:, :, 2]

        color_difference = (
            np.mean(np.abs(red - green))
            + np.mean(np.abs(green - blue))
            + np.mean(np.abs(red - blue))
        ) / 3.0

        if color_difference > 25:
            return False, (
                "Please enter a correct lung CT scan image. "
                "The uploaded image does not appear to be "
                "a grayscale medical CT image."
            )

        gray = np.asarray(
            image.convert("L"),
            dtype=np.float32,
        )

        if np.std(gray) < 8:
            return False, (
                "Please enter a correct lung CT scan image. "
                "The uploaded image appears to be blank."
            )

        if np.max(gray) - np.min(gray) < 15:
            return False, (
                "Please enter a correct lung CT scan image."
            )

        return True, None

    except Exception:
        logger.exception(
            "Input image validation failed."
        )

        return False, (
            "Unable to validate the uploaded image. "
            "Please upload a valid lung CT scan."
        )


def load_class_names():
    global _class_names

    if _class_names is not None:
        return _class_names

    if not CLASSES_PATH.exists():
        logger.warning(
            "classes.txt not found. Using default labels."
        )

        _class_names = [
            "Class 0",
            "Class 1",
        ]

        return _class_names

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        classes = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not classes:
        raise RuntimeError(
            "classes.txt is empty."
        )

    _class_names = classes

    return _class_names


def load_unet():
    global _unet_model

    if _unet_model is not None:
        return _unet_model

    if not UNET_PATH.exists():
        raise FileNotFoundError(
            f"U-Net model not found: {UNET_PATH}"
        )

    logger.info(
        "Loading U-Net model from %s",
        UNET_PATH,
    )

    gc.collect()

    try:
        _unet_model = load_model(
            UNET_PATH,
            compile=False,
        )

        logger.info(
            "U-Net model loaded successfully."
        )

        return _unet_model

    except Exception:
        logger.exception(
            "Failed to load U-Net model."
        )

        _unet_model = None
        gc.collect()

        raise


def load_classifier():
    global _classifier_model

    if _classifier_model is not None:
        return _classifier_model

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {CLASSIFIER_PATH}"
        )

    classes = load_class_names()

    logger.info(
        "Loading classifier model from %s",
        CLASSIFIER_PATH,
    )

    model = CapsuleNetwork(
        num_classes=len(classes)
    )

    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint[
                "model_state_dict"
            ]

    model.load_state_dict(
        checkpoint,
        strict=False,
    )

    model.to(DEVICE)

    model.eval()

    _classifier_model = model

    return _classifier_model


def prepare_image(image):
    """
    Prepare the uploaded image for the trained U-Net.

    The saved U-Net expects RGB input with shape:
        (batch, 256, 256, 3)

    The downstream classifier is converted to grayscale
    separately after U-Net ROI extraction.
    """

    image = image.convert("RGB")

    original_width, original_height = image.size

    resized = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        resized,
        dtype=np.float32,
    ) / 255.0

    # Add batch dimension.
    # Final shape: (1, 256, 256, 3)
    array = np.expand_dims(
        array,
        axis=0,
    )

    return (
        array,
        original_width,
        original_height,
    )


def generate_segmentation(image_array):
    model = load_unet()

    if image_array.ndim != 4 or image_array.shape[-1] != 3:
        raise ValueError(
            "Internal image preprocessing error: "
            "U-Net requires RGB input with shape "
            "(1, 256, 256, 3)."
        )

    prediction = model.predict(
        image_array,
        verbose=0,
    )

    mask = prediction[0]

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    binary_mask = (
        mask >= 0.5
    ).astype(np.uint8)

    return binary_mask


def extract_roi(image_array, mask):
    """
    Extract the segmented ROI from the RGB U-Net input.

    The classifier expects a single grayscale channel, so the
    RGB image is converted to grayscale only after segmentation.
    """

    rgb_image = image_array[0]

    # RGB -> grayscale while keeping values in [0, 1].
    image = (
        0.299 * rgb_image[:, :, 0]
        + 0.587 * rgb_image[:, :, 1]
        + 0.114 * rgb_image[:, :, 2]
    )

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return image

    x_min = max(
        int(xs.min()),
        0,
    )

    x_max = min(
        int(xs.max()) + 1,
        image.shape[1],
    )

    y_min = max(
        int(ys.min()),
        0,
    )

    y_max = min(
        int(ys.max()) + 1,
        image.shape[0],
    )

    roi = image[
        y_min:y_max,
        x_min:x_max,
    ]

    if roi.size == 0:
        return image

    return roi


def prepare_classifier_input(roi):
    image = Image.fromarray(
        np.uint8(
            np.clip(roi, 0, 1) * 255
        )
    )

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    tensor = torch.from_numpy(
        array
    ).unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE)


def classify_roi(roi):
    model = load_classifier()

    classes = load_class_names()

    tensor = prepare_classifier_input(
        roi
    )

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    predicted_index = int(
        prediction.item()
    )

    confidence_value = float(
        confidence.item()
    )

    if predicted_index >= len(classes):
        predicted_class = (
            f"Class {predicted_index}"
        )
    else:
        predicted_class = classes[
            predicted_index
        ]

    probs = []

    probability_values = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    for index, probability in enumerate(
        probability_values
    ):
        if index < len(classes):
            class_name = classes[index]
        else:
            class_name = f"Class {index}"

        probs.append(
            {
                "name": class_name,
                "prob": float(probability),
            }
        )

    return {
        "label": predicted_class,
        "confidence": confidence_value,
        "confidence_percent": round(
            confidence_value * 100,
            2,
        ),
        "probs": probs,
    }


def run_pipeline(image):

    # ========================================================
    # STEP 1: Basic image validation
    # ========================================================

    is_valid, validation_error = validate_input_image(image)

    if not is_valid:
        raise ValueError(validation_error)

    # ========================================================
    # STEP 2: Prepare RGB image for U-Net
    # ========================================================

    image_array, width, height = prepare_image(image)

    # ========================================================
    # STEP 3: U-Net segmentation
    # ========================================================

    mask = generate_segmentation(image_array)

    # ========================================================
    # STEP 4: Validate lung segmentation
    # ========================================================

    mask_pixels = int(np.sum(mask))

    total_pixels = int(
        mask.shape[0] * mask.shape[1]
    )

    mask_ratio = (
        mask_pixels /
        max(total_pixels, 1)
    )

    logger.info(
        "U-Net mask ratio: %.4f",
        mask_ratio
    )

    # No lung region detected
    if mask_pixels == 0:
        raise ValueError(
            "Invalid image. "
            "Please upload a valid lung CT scan image."
        )

    # Very small segmentation is unlikely to be a valid lung ROI
    if mask_ratio < 0.03:
        raise ValueError(
            "Invalid image. "
            "No valid lung region was detected. "
            "Please upload a lung CT scan."
        )

    # A mask covering almost the entire image is suspicious
    if mask_ratio > 0.70:
        raise ValueError(
            "Invalid image. "
            "The uploaded image does not appear to "
            "contain a valid lung CT region."
        )

    # ========================================================
    # STEP 5: Extract lung ROI
    # ========================================================

    roi = extract_roi(
        image_array,
        mask
    )

    # ========================================================
    # STEP 6: Classify only after validation
    # ========================================================

    result = classify_roi(
        roi
    )

    # ========================================================
    # STEP 7: Metadata
    # ========================================================

    result["input_width"] = width
    result["input_height"] = height
    result["segmentation_detected"] = True
    result["mask_ratio"] = round(
        mask_ratio,
        4
    )

    return result


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(
    RequestEntityTooLarge
)
def handle_file_too_large(error):
    return (
        jsonify(
            {
                "error": (
                    f"File exceeds the "
                    f"{MAX_UPLOAD_MB} MB limit."
                )
            }
        ),
        413,
    )


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html"
    )


@app.route("/predict")
def predict_page():
    return render_template(
        "predict.html"
    )


@app.route("/health")
def health():
    # Keep health checks lightweight; never load ML models here.
    return jsonify(
        {
            "status": "healthy",
            "service": "LungVision AI",
            "device": str(DEVICE),
            "models_loaded": {
                "unet": _unet_model is not None,
                "classifier": _classifier_model is not None,
            },
        }
    )


@app.route("/api/health")
def api_health():
    # Check model files without loading them.
    return jsonify(
        {
            "status": "ok",
            "service": "LungVision AI",
            "device": str(DEVICE),
            "models": {
                "unet_file": UNET_PATH.exists(),
                "classifier_file": CLASSIFIER_PATH.exists(),
                "classes_file": CLASSES_PATH.exists(),
                "unet_loaded": _unet_model is not None,
                "classifier_loaded": _classifier_model is not None,
            },
        }
    )


@app.route(
    "/api/predict",
    methods=["POST"],
)
def api_predict():
    if "file" not in request.files:
        return jsonify(
            {
                "success": False,
                "error": "No image file supplied.",
            }
        ), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify(
            {
                "success": False,
                "error": "Filename is empty.",
            }
        ), 400

    safe_name = secure_filename(
        uploaded_file.filename
    )

    if not allowed_file(safe_name):
        return jsonify(
            {
                "success": False,
                "error": (
                    "Unsupported file type. "
                    "Use PNG, JPG or JPEG."
                ),
            }
        ), 400

    try:
        image_bytes = uploaded_file.read()

        if not image_bytes:
            return jsonify(
                {
                    "success": False,
                    "error": "Uploaded file is empty.",
                }
            ), 400

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        result = run_pipeline(
            image
        )

        return jsonify(
            {
                "success": True,
                "filename": safe_name,
                "prediction": result,
            }
        )

    except UnidentifiedImageError:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Uploaded file is not a valid image."
                ),
            }
        ), 400

    except ValueError as error:
        logger.warning(
            "API image validation rejected: %s",
            error,
        )

        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 400

    except FileNotFoundError as error:
        logger.exception(
            "Model file missing."
        )

        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 500

    except Exception:
        logger.exception(
            "Prediction failed."
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Prediction failed. "
                    "Check server logs."
                ),
            }
        ), 500


@app.route(
    "/predict",
    methods=["POST"],
)
def predict():
    if "image" not in request.files:
        return render_template(
            "result.html",
            error=(
                "Please upload a lung CT scan image."
            ),
        )

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return render_template(
            "result.html",
            error="Please select an image.",
        )

    safe_name = secure_filename(
        uploaded_file.filename
    )

    if not allowed_file(safe_name):
        return render_template(
            "result.html",
            error=(
                "Invalid file type. "
                "Please upload a JPG, JPEG or PNG "
                "lung CT scan image."
            ),
        )

    try:
        image_bytes = uploaded_file.read()

        if not image_bytes:
            return render_template(
                "result.html",
                error=(
                    "The uploaded file is empty. "
                    "Please upload a valid lung CT scan."
                ),
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        result = run_pipeline(
            image
        )

        predicted_class = result.get(
            "label",
            "Unknown",
        )

        confidence = result.get(
            "confidence_percent",
            0,
        )

        probs = result.get(
            "probs",
            [],
        )

        return render_template(
            "result.html",
            predicted_class=predicted_class,
            confidence=confidence,
            probs=probs,
            filename=safe_name,
            result=result,
        )

    except UnidentifiedImageError:
        logger.warning(
            "Invalid image uploaded: %s",
            safe_name,
        )

        return render_template(
            "result.html",
            error=(
                "Please enter a correct lung CT scan image. "
                "The uploaded file is not a valid image."
            ),
        )

    except ValueError as error:
        logger.warning(
            "Image validation rejected: %s",
            error,
        )

        return render_template(
            "result.html",
            error=str(error),
        )

    except FileNotFoundError:
        logger.exception(
            "Required model file is missing."
        )

        return render_template(
            "result.html",
            error=(
                "Required AI model files are missing. "
                "Please check the models directory."
            ),
        )

    except Exception:
        logger.exception(
            "Unexpected prediction error."
        )

        return render_template(
            "result.html",
            error=(
                "Prediction could not be completed. "
                "Please try another valid lung CT image."
            ),
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv("PORT", "5000")
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
