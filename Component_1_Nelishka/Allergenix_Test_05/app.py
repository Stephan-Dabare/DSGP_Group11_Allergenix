from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import torch
from ultralytics import YOLO
import pytesseract
from collections import OrderedDict
import pandas as pd
from fuzzywuzzy import process
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
model = YOLO('best.pt')

dataset_path = r"C:\Users\nelis\Desktop\Ingredient_cleaned.csv"
dataset = pd.read_csv(dataset_path)
valid_ingredients = dataset['ingredient'].str.lower().str.strip().tolist()

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/component_1')
def component_1():
    return render_template('component_1.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Ensure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        ingredients, img_path = process_image(filepath)
        return render_template('component_1.html', ingredients=ingredients, img_path=img_path)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[^a-zA-Z0-9, ]+', '', text)
    words = re.split(r'[;,/\+]', text)
    return [word.strip() for word in words if word.strip()]

def process_image(img_path):
    img = cv2.imread(img_path)
    results = model(img)

    ingredient_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            custom_config = r'--oem 3 --psm 6 -l eng'
            extracted_text = pytesseract.image_to_string(cropped_img, config=custom_config)
            if extracted_text:
                cleaned_words = clean_text(extracted_text)
                for word in cleaned_words:
                    ingredient_boxes.append((y1, x1, word))

    ingredient_boxes.sort(key=lambda b: (round(b[0] / 10), b[1]))
    detected_ingredients = list(OrderedDict.fromkeys([text for (_, _, text) in ingredient_boxes]))

    matched_ingredients = []
    MATCH_THRESHOLD = 70
    for ingredient in detected_ingredients:
        match, score = process.extractOne(ingredient, valid_ingredients)
        if score >= MATCH_THRESHOLD:
            matched_ingredients.append(match)
    matched_ingredients = list(OrderedDict.fromkeys(matched_ingredients))

    if matched_ingredients:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + os.path.basename(img_path))
    cv2.imwrite(output_img_path, img)

    return matched_ingredients, 'uploads/' + os.path.basename(output_img_path)

if __name__ == '__main__':
    app.run(debug=True)