from flask import Flask, render_template, request, jsonify
import os
import cv2
import re
import torch
import easyocr
from ultralytics import YOLO
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


model = YOLO(
    r"F:\University\2_Year_02\2_Year_02_Sem1\0_Data_Science\Component_1_Nelishka\Yolo_11_x\runs\detect\train\weights\best.pt")


reader = easyocr.Reader(['en'])

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def detect_and_extract_ingredients(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Run YOLO detection
    results = model(image)

    ingredient_texts = []

    for result in results:
        for box in result.boxes.xyxy:  # Get bounding boxes
            x1, y1, x2, y2 = map(int, box)
            cropped_img = image[y1:y2, x1:x2]  # Crop detected region

            # Convert to grayscale for better OCR performance
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            # OCR text extraction
            extracted_text = reader.readtext(gray, detail=0)

            # Join lines, clean up text, and split by commas
            cleaned_text = ' '.join(extracted_text).lower()
            ingredients = re.split(r',|\n', cleaned_text)  # Split by commas or newlines
            ingredients = [ing.strip() for ing in ingredients if ing.strip()]  # Remove empty entries

            ingredient_texts.extend(ingredients)

    return ingredient_texts


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Detect and extract ingredients
        extracted_ingredients = detect_and_extract_ingredients(filepath)
        return jsonify({'filepath': filepath, 'results': {'ingredients': extracted_ingredients}})

    return jsonify({'error': 'Invalid file type'})


if __name__ == '__main__':
    app.run(debug=True)