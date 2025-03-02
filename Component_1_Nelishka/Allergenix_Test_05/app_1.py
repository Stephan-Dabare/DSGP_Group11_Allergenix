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

class ImageProcessor:
    def __init__(self, model_path, upload_folder):
        self.model = YOLO(model_path)
        self.upload_folder = upload_folder

    def process_image(self, img_path):
        img = cv2.imread(img_path)
        results = self.model(img)
        ingredient_boxes = self.extract_ingredient_boxes(results, img)
        detected_ingredients = self.get_detected_ingredients(ingredient_boxes)
        output_img_path = self.save_processed_image(img, img_path, results)
        return detected_ingredients, output_img_path

    def extract_ingredient_boxes(self, results, img):
        ingredient_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_img = img[y1:y2, x1:x2]
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                custom_config = r'--oem 3 --psm 6 -l eng'
                extracted_text = pytesseract.image_to_string(cropped_img, config=custom_config)
                if extracted_text:
                    cleaned_words = self.clean_text(extracted_text)
                    for word in cleaned_words:
                        ingredient_boxes.append((y1, x1, word))
        return ingredient_boxes

    @staticmethod
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'[^a-zA-Z0-9, ]+', '', text)
        words = re.split(r'[;,/\+]', text)
        return [word.strip() for word in words if word.strip()]

    @staticmethod
    def get_detected_ingredients(ingredient_boxes):
        ingredient_boxes.sort(key=lambda b: (round(b[0] / 10), b[1]))
        return list(OrderedDict.fromkeys([text for (_, _, text) in ingredient_boxes]))

    def save_processed_image(self, img, img_path, results):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        output_img_path = os.path.join(self.upload_folder, 'output_' + os.path.basename(img_path))
        cv2.imwrite(output_img_path, img)
        return 'uploads/' + os.path.basename(output_img_path)

class IngredientMatcher:
    def __init__(self, dataset_path, match_threshold=70):
        self.dataset = pd.read_csv(dataset_path)
        self.valid_ingredients = self.dataset['ingredient'].str.lower().str.strip().tolist()
        self.match_threshold = match_threshold

    def match_ingredients(self, detected_ingredients):
        matched_ingredients = []
        for ingredient in detected_ingredients:
            match, score = process.extractOne(ingredient, self.valid_ingredients)
            if score >= self.match_threshold:
                matched_ingredients.append(match)
        return list(OrderedDict.fromkeys(matched_ingredients))

class FlaskApp:
    def __init__(self, upload_folder, model_path, dataset_path):
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = upload_folder
        self.image_processor = ImageProcessor(model_path, upload_folder)
        self.ingredient_matcher = IngredientMatcher(dataset_path)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('home.html')

        @self.app.route('/component_1')
        def component_1():
            return render_template('component_1.html')

        @self.app.route('/upload', methods=['POST'])
        def upload():
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file:
                if not os.path.exists(self.app.config['UPLOAD_FOLDER']):
                    os.makedirs(self.app.config['UPLOAD_FOLDER'])
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                ingredients, img_path = self.image_processor.process_image(filepath)
                matched_ingredients = self.ingredient_matcher.match_ingredients(ingredients)
                return render_template('component_1.html', ingredients=matched_ingredients, img_path=img_path)

    def run(self):
        self.app.run(debug=True)


if __name__ == '__main__':
    upload_folder = 'static/uploads/'
    model_path = 'best.pt'
    dataset_path = r"C:\Users\nelis\Desktop\Ingredient_cleaned.csv"
    flask_app = FlaskApp(upload_folder, model_path, dataset_path)
    flask_app.run()