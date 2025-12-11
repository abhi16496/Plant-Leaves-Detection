import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
import uuid

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and metadata
MODEL_PATH = "plant_disease_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
CONFIG_PATH = "model_config.pkl"

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)

# Feature extraction
def extract_features(image):
    image = image.resize((config['image_width'], config['image_height']))
    image_np = np.array(image.convert('RGB'))[:, :, ::-1].copy()  # RGB to BGR
    features = []

    if config['use_color_histogram']:
        for i in range(3):
            hist = cv2.calcHist([image_np], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist)
            features.extend(hist.flatten())

    if config['use_texture_features']:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            features.append(graycoprops(glcm, prop)[0, 0])

    if config['use_mean_std']:
        mean, std_dev = cv2.meanStdDev(image_np)
        features.extend(mean.flatten())
        features.extend(std_dev.flatten())

    return np.array(features).reshape(1, -1)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save uploaded image
    unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(image_path)

    # Open image and extract features
    image = Image.open(image_path)
    features = extract_features(image)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    # Return prediction and image URL
    return jsonify({
        'label': predicted_label,
        'image_url': url_for('static', filename=f'uploads/{unique_filename}')
    })

if __name__ == '__main__':
    app.run(debug=True)
