import os
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename


MODEL_PATH = os.getenv("MODEL_PATH", "final_model.keras")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", tempfile.gettempdir())) / "pneumoscan_uploads"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "16"))
TARGET_SIZE = (150, 150)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
model = load_model(MODEL_PATH)


def preprocess_image(image_path: Path):
    img = load_img(image_path, target_size=TARGET_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_image(image_path: Path) -> str:
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image, verbose=0)
    return "Pneumonia" if prediction[0][0] > 0.5 else "Normal"


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
        result = predict_image(image_path)
        return jsonify({"prediction": result}), 200
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "detail": str(exc)}), 500
    finally:
        image_path.unlink(missing_ok=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
