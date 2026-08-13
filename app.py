import os
import uuid

import torch
import torch.nn as nn

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
MODEL_PATH = "models/RS_CapsNet.pth"
CLASS_PATH = "models/classes.txt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# MODEL
# ============================================================

class RS_CapsNet(nn.Module):

    def __init__(self, num_classes=3):
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

print("Classes:", class_names)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "models/RS_CapsNet.pth not found."
    )

model = RS_CapsNet(
    num_classes=len(class_names)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# PREDICTION PAGE + PREDICTION
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # --------------------------------------------------------
    # GET REQUEST
    # Open prediction page
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template("predict.html")


    # --------------------------------------------------------
    # POST REQUEST
    # Process uploaded image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400


    file = request.files["image"]


    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if file.filename == "":

        return jsonify({
            "error": "Empty filename"
        }), 400


    # --------------------------------------------------------
    # Check file extension
    # --------------------------------------------------------

    if not file.filename.lower().endswith(
        (".jpg", ".jpeg")
    ):

        return jsonify({
            "error": "Only JPEG images (.jpg/.jpeg) are allowed."
        }), 400


    # --------------------------------------------------------
    # Generate unique filename
    # --------------------------------------------------------

    filename = (
        uuid.uuid4().hex
        + "_"
        + file.filename
    )


    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    file.save(filepath)


    try:

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            filepath
        ).convert("L")


        # ----------------------------------------------------
        # Transform image
        # ----------------------------------------------------

        input_tensor = transform(image)


        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        input_tensor = input_tensor.unsqueeze(0)


        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        input_tensor = input_tensor.to(device)


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]


        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predicted_index = int(
            probabilities.argmax().item()
        )


        predicted_class = class_names[
            predicted_index
        ]


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = (
            float(
                probabilities[
                    predicted_index
                ].item()
            )
            * 100
        )


        # ----------------------------------------------------
        # All probabilities
        # ----------------------------------------------------

        probs = []

        for i in range(
            len(class_names)
        ):

            probs.append({
                "name": class_names[i],
                "prob": float(
                    probabilities[i].item()
                )
            })


        # ----------------------------------------------------
        # Result page
        # ----------------------------------------------------

        return render_template(
            "result.html",
            predicted_class=predicted_class,
            confidence=confidence,
            probs=probs
        )


    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "model": "loaded",
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