from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os    

app = Flask(__name__)

# Load the model
model_path = 'final_model.keras'
model = load_model(model_path)

def preprocess_image(image, target_size):
    try:
        img = load_img(image, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None

def predict(image):
    preprocessed_image = preprocess_image(image, target_size=(150, 150))
    if preprocessed_image is not None:
        prediction = model.predict(preprocessed_image)
        class_label = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
        return class_label
    else:
        return "Error in image preprocessing"

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image_path = os.path.join('temp', file.filename)
        file.save(image_path)
        result = predict(image_path)
        os.remove(image_path)
        return jsonify({'prediction': result}), 200

    return jsonify({'error': 'File not processed'}), 400

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
