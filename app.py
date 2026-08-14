import os
import gc
import uuid

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import cv2
import numpy as npgit 
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "models/RS_CapsNet.pth"
CLASS_PATH = "models/classes.txt"
UNET_PATH = "models/final_unet.keras"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Render free/small instances: force CPU and keep PyTorch thread count low.
device = torch.device("cpu")
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

MIN_CLASS_CONFIDENCE = 60.0
MIN_LUNG_AREA_RATIO = 0.03
MAX_LUNG_AREA_RATIO = 0.85


class RS_CapsNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# -----------------------------
# Class names
# -----------------------------
if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(f"{CLASS_PATH} not found.")

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

if not class_names:
    raise ValueError("No classes found in models/classes.txt.")


# -----------------------------
# Load ONLY PyTorch classifier at startup
# U-Net is lazy-loaded during prediction to reduce peak startup RAM.
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found.")

model = RS_CapsNet(num_classes=len(class_names))

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(state_dict)
del state_dict

model = model.to(device)
model.eval()

print("Using device:", device)
print("Classes:", class_names)
print("RS-CapsNet loaded successfully.")


# Classifier preprocessing must match training.
classifier_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# -----------------------------
# Lazy U-Net loading
# -----------------------------
_unet = None


def load_unet():
    global _unet

    if _unet is not None:
        return _unet

    if not os.path.exists(UNET_PATH):
        raise FileNotFoundError(
            "models/final_unet.keras not found. "
            "The U-Net model is required for input validation."
        )

    # Import TensorFlow only when prediction actually needs it.
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    _unet = tf.keras.models.load_model(
        UNET_PATH,
        compile=False
    )

    print("U-Net loaded for prediction.")
    return _unet


def unload_unet():
    global _unet

    if _unet is not None:
        del _unet
        _unet = None

    gc.collect()

    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass


# -----------------------------
# Basic input validation
# -----------------------------
def looks_like_lung_ct(image_bgr):
    """
    Conservative gate for obvious non-CT photographs.

    This is NOT a medical diagnosis and cannot prove that an image
    is a CT scan. It only rejects obvious color photographs.
    """
    if image_bgr is None or image_bgr.size == 0:
        return False, "Invalid image."

    h, w = image_bgr.shape[:2]

    if h < 64 or w < 64:
        return False, "Image is too small."

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Family photos/selfies are commonly strongly colored.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))

    # Keep this conservative because some CT images may have annotations.
    if saturation > 55:
        return False, (
            "This does not look like a grayscale lung CT image. "
            "Please upload a lung CT JPEG image."
        )

    # Reject almost completely uniform images.
    if float(np.std(gray)) < 8:
        return False, "Image has insufficient visual information."

    return True, None


# -----------------------------
# U-Net segmentation
# -----------------------------
def get_lung_roi(image_bgr):
    unet = load_unet()

    resized = cv2.resize(
        image_bgr,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    inp = resized.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)

    pred = unet.predict(inp, verbose=0)[0, :, :, 0]

    del inp, resized

    mask = (pred > 0.5).astype(np.uint8) * 255

    # Clean small noise.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the largest connected region.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    image_area = float(mask.shape[0] * mask.shape[1])
    area_ratio = area / image_area

    if area_ratio < MIN_LUNG_AREA_RATIO or area_ratio > MAX_LUNG_AREA_RATIO:
        return None, area_ratio

    x, y, w, h = cv2.boundingRect(largest)

    # Apply the mask to the ROI.
    full_mask = cv2.resize(
        mask,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    roi = image_bgr[y:y+h, x:x+w] if (
        y + h <= image_bgr.shape[0] and x + w <= image_bgr.shape[1]
    ) else image_bgr

    roi_mask = full_mask[y:y+h, x:x+w]

    if roi.size == 0 or roi_mask.size == 0:
        return None, area_ratio

    roi = cv2.bitwise_and(roi, roi, mask=roi_mask)

    # If ROI contains almost no pixels, reject it.
    if np.count_nonzero(roi_mask) < 100:
        return None, area_ratio

    return roi, area_ratio


def classify_roi(roi):
    pil_image = Image.fromarray(
        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    )

    input_tensor = classifier_transform(pil_image).unsqueeze(0)

    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item() * 100)

    probs = [
        {
            "name": class_names[i],
            "prob": round(float(probabilities[i].item()), 4)
        }
        for i in range(len(class_names))
    ]

    return class_names[predicted_index], confidence, probs


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    filepath = None

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files["image"]

        if not file.filename:
            return jsonify({"error": "No file selected."}), 400

        original_filename = file.filename.lower()

        if not original_filename.endswith((".jpg", ".jpeg")):
            return jsonify({
                "error": "Only JPG and JPEG images are allowed."
            }), 400

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

        # Read with OpenCV for validation and U-Net.
        image_bgr = cv2.imread(filepath)

        if image_bgr is None:
            return jsonify({
                "error": "Could not read the uploaded image."
            }), 400

        # Layer 1: obvious non-CT rejection.
        valid, reason = looks_like_lung_ct(image_bgr)

        if not valid:
            return render_template(
                "result.html",
                predicted_class="Invalid Input",
                confidence=0,
                probs=[],
                error_message=reason
            )

        # Layer 2: U-Net lung-region validation.
        roi, lung_area_ratio = get_lung_roi(image_bgr)

        if roi is None:
            return render_template(
                "result.html",
                predicted_class="Invalid Input",
                confidence=0,
                probs=[],
                error_message=(
                    "No valid lung region was detected. "
                    "Please upload a clear lung CT image."
                )
            )

        # Layer 3: classifier.
        predicted_class, confidence, probs = classify_roi(roi)

        if confidence < MIN_CLASS_CONFIDENCE:
            return render_template(
                "result.html",
                predicted_class="Uncertain",
                confidence=confidence,
                probs=probs,
                error_message=(
                    "The model is not confident enough to classify this image."
                )
            )

        return render_template(
            "result.html",
            predicted_class=predicted_class,
            confidence=confidence,
            probs=probs
        )

    except Exception as e:
        print("Prediction error:", repr(e))

        return jsonify({
            "error": "Prediction failed.",
            "details": str(e)
        }), 500

    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as cleanup_error:
                print("Could not remove temporary image:", cleanup_error)

        # Release temporary TensorFlow memory after every prediction.
        unload_unet()


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "loaded",
        "device": str(device),
        "classes": class_names
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
