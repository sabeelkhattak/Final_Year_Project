import os
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename


MODEL_PATH = os.getenv("MODEL_PATH", "final_model.keras")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", tempfile.gettempdir())) / "pneumoscan_uploads"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "16"))
TARGET_SIZE = (150, 150)
MIN_IMAGE_SIDE = int(os.getenv("MIN_IMAGE_SIDE", "100"))
MAX_AVERAGE_SATURATION = float(os.getenv("MAX_AVERAGE_SATURATION", "0.22"))
MAX_HIGH_SATURATION_RATIO = float(os.getenv("MAX_HIGH_SATURATION_RATIO", "0.18"))
MAX_COLOR_CHANNEL_SPREAD = float(os.getenv("MAX_COLOR_CHANNEL_SPREAD", "0.12"))
MIN_GRAYSCALE_CONTRAST = float(os.getenv("MIN_GRAYSCALE_CONTRAST", "0.06"))
MAX_GRAYSCALE_MEAN = float(os.getenv("MAX_GRAYSCALE_MEAN", "0.85"))
MIN_DARK_PIXEL_RATIO = float(os.getenv("MIN_DARK_PIXEL_RATIO", "0.08"))
NON_XRAY_MESSAGE = "Image is not a chest X-ray. Please capture a clear lung X-ray image."

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
model = load_model(MODEL_PATH)


def preprocess_image(image_path: Path):
    img = load_img(image_path, target_size=TARGET_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def is_probably_lung_xray(image_path: Path) -> bool:
    """Reject obvious non-X-ray images before sending them to the binary model."""
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
                return False

            image.thumbnail((256, 256))
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError):
        return False

    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = np.divide(
        max_channel - min_channel,
        max_channel,
        out=np.zeros_like(max_channel),
        where=max_channel > 0.0,
    )
    grayscale = rgb.mean(axis=2)
    channel_spread = np.mean(
        np.abs(rgb[:, :, 0] - rgb[:, :, 1])
        + np.abs(rgb[:, :, 1] - rgb[:, :, 2])
        + np.abs(rgb[:, :, 0] - rgb[:, :, 2])
    ) / 3.0

    average_saturation = float(saturation.mean())
    high_saturation_ratio = float((saturation > 0.35).mean())
    grayscale_mean = float(grayscale.mean())
    grayscale_contrast = float(grayscale.std())
    dark_pixel_ratio = float((grayscale < 0.25).mean())

    if grayscale_contrast < MIN_GRAYSCALE_CONTRAST:
        return False

    very_colorful = (
        average_saturation > MAX_AVERAGE_SATURATION
        and high_saturation_ratio > MAX_HIGH_SATURATION_RATIO
        and channel_spread > MAX_COLOR_CHANNEL_SPREAD
    )
    very_bright_flat = grayscale_mean > MAX_GRAYSCALE_MEAN and dark_pixel_ratio < MIN_DARK_PIXEL_RATIO

    return not (very_colorful or very_bright_flat)


def predict_image(image_path: Path) -> tuple[str, float]:
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image, verbose=0)
    probability = float(prediction[0][0])
    result = "Pneumonia" if probability > 0.5 else "Normal"
    return result, probability


@app.get("/")
def index():
    return jsonify(
        {
            "service": "PneumoScan API",
            "status": "ok",
            "predict_route": "/predict",
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(uploaded_file.filename) or "upload.jpg"
    image_path = UPLOAD_DIR / filename

    try:
        uploaded_file.save(image_path)
        if not is_probably_lung_xray(image_path):
            return jsonify({"error": "invalid_image", "message": NON_XRAY_MESSAGE}), 422

        result, probability = predict_image(image_path)
        return jsonify({"prediction": result, "probability": probability}), 200
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "prediction_failed", "message": "Prediction failed", "detail": str(exc)}), 500
    finally:
        image_path.unlink(missing_ok=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
