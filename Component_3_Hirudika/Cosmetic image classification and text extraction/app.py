import os
import re
import cv2
import numpy as np
import joblib
import torch
import pytesseract
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.applications.efficientnet import preprocess_input
from transformers import BertForSequenceClassification, AutoTokenizer
from PIL import ImageFile
from flask import Flask, request, jsonify, render_template
from PIL import Image
import base64
import io

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow messages

app = Flask(__name__)

# Load EfficientNetV2L as feature extractor
base_model = EfficientNetV2L(weights='imagenet', include_top=False, pooling='avg')

# Load One-Class SVM and preprocessing models
pca = joblib.load(r"C:\Users\LENOVO\Desktop\Final\Final\models\CNN\pca_model.pkl")
scaler = joblib.load(r"C:\Users\LENOVO\Desktop\Final\Final\models\CNN\scaler.pkl")
minmax_scaler = joblib.load(r"C:\Users\LENOVO\Desktop\Final\Final\models\CNN\minmax_scaler.pkl")
oc_svm = joblib.load(r"C:\Users\LENOVO\Desktop\Final\Final\models\CNN\ocsvm_model.pkl")

# Load fine-tuned BERT models and tokenizers
model_path_1 = r"C:\Users\LENOVO\Desktop\fine_tuned_bert_____"
model_path_2 = r"C:\Users\LENOVO\Desktop\fine_tuned_bert___"

try:
    model_1 = BertForSequenceClassification.from_pretrained(model_path_1)
    tokenizer_1 = AutoTokenizer.from_pretrained(model_path_1)
    model_2 = BertForSequenceClassification.from_pretrained(model_path_2)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_path_2)
except Exception as e:
    print(f"Error loading models: {e}")

# Keywords to look for in the extracted text
KEYWORDS = ["key ingredients", "key ingredient", "ingredient", "ingredients", "content", "component", "composition"]


# Image classification function
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def classify_image(image_path):
    try:
        img_array = preprocess_image(image_path)
        feature = base_model.predict(img_array)
        feature = np.array(feature, dtype=np.float64)

        feature_scaled = scaler.transform(feature)
        feature_pca = pca.transform(feature_scaled)
        feature_normalized = minmax_scaler.transform(feature_pca)

        prediction = oc_svm.predict(feature_normalized)
        return "Normal" if prediction[0] == 1 else "Anomaly"
    except Exception as e:
        return f"Error: {str(e)}"


# Function to preprocess image for OCR
def preprocess_image_ocr(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(100, 100))
    enhanced = clahe.apply(denoised)
    return enhanced


# Function to clean extracted text
def clean_extracted_text(text):
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = re.sub(r"[^a-zA-Z0-9\s,\-%/]", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        words = line.split()
        if len(words) > 0 and len(words[0]) <= 2 and words[0].isalpha():
            words = words[1:]
        if len(words) > 0 and len(words[-1]) <= 2 and words[-1].isalpha():
            words = words[:-1]
        cleaned_line = " ".join(words)
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    return cleaned_lines


# Function to classify an ingredient using both models
def is_cosmetic_ingredient(ingredient):
    try:
        inputs_1 = tokenizer_1(ingredient, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs_2 = tokenizer_2(ingredient, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs_1 = model_1(**inputs_1)
            outputs_2 = model_2(**inputs_2)
        probs_1 = torch.nn.functional.softmax(outputs_1.logits, dim=-1).cpu().numpy()
        probs_2 = torch.nn.functional.softmax(outputs_2.logits, dim=-1).cpu().numpy()
        avg_probs = (probs_1 + probs_2) / 2.0
        predicted_label = np.argmax(avg_probs, axis=1)[0]
        return predicted_label == 1
    except Exception as e:
        print(f"Error in ingredient classification: {e}")
        return False


# Function to extract and filter cosmetic ingredients
def extract_cosmetic_ingredients(image):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        processed_image = preprocess_image_ocr(image)
        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(processed_image, config=custom_config)
        ingredient_lines = clean_extracted_text(extracted_text)
        if not ingredient_lines:
            return []

        keyword_index = -1
        for i, line in enumerate(ingredient_lines):
            if any(keyword.lower() in line.lower() for keyword in KEYWORDS):
                keyword_index = i
                break

        relevant_lines = ingredient_lines[keyword_index:] if keyword_index != -1 else ingredient_lines
        cosmetic_ingredients = [line for line in relevant_lines if is_cosmetic_ingredient(line)]
        return cosmetic_ingredients
    except Exception as e:
        print(f"Error in extract_cosmetic_ingredients: {e}")
        return []


@app.route('/')
def index():
    return render_template('header.html')  # Removed incorrect "templates/"


@app.route('/extract_text', methods=['POST'])
def extract_text():
    try:
        image_data = request.json['image']
        image_data = image_data.split(',')[1]  # Remove the data URL prefix
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Step 1: Classify as Normal/Anomaly
        classification = classify_image(image)

        if classification == "Non-Cosmetic":
            return jsonify({'status': 'success', 'classification': classification, 'text': []})

        # Step 2: Extract and classify ingredients
        cosmetic_ingredients = extract_cosmetic_ingredients(image)

        # Step 3: Process ingredient list
        if not cosmetic_ingredients:
            return jsonify({'status': 'success', 'classification': "Non-Cosmetic", 'text': []})

        processed_ingredients = []
        temp = ""
        for line in cosmetic_ingredients:
            line = line.strip()
            if not line.endswith(","):
                temp += " " + line
            else:
                temp += " " + line
                processed_ingredients.append(temp.strip())
                temp = ""

        if temp:
            processed_ingredients.append(temp.strip())

        cleaned_ingredients = []
        for line in processed_ingredients:
            parts = [ingredient.strip() for ingredient in line.split(',') if ingredient.strip()]
            parts = [ingredient for ingredient in parts if not ingredient.replace(".", "").isdigit()]
            parts = [re.sub(r"[^a-zA-Z0-9\s\-%]", "", ingredient) for ingredient in parts]

            cleaned_parts = []
            for ingredient in parts:
                words = ingredient.split()
                if words and words[0].isdigit():
                    words.pop(0)
                while words and len(words[-1]) <= 2 and words[-1].isalpha():
                    words.pop()

                cleaned_ingredient = " ".join(words)
                for keyword in KEYWORDS:
                    cleaned_ingredient = re.sub(rf"\b{re.escape(keyword)}\b", "", cleaned_ingredient,
                                                flags=re.IGNORECASE).strip()
                if cleaned_ingredient:
                    cleaned_parts.append(cleaned_ingredient)

            cleaned_ingredients.extend(cleaned_parts)

        final_classification = "Cosmetic" if cleaned_ingredients else "Non-Cosmetic"
        return jsonify({'status': 'success', 'classification': final_classification, 'text': cleaned_ingredients})
    except Exception as e:
        print(f"Error in extract_text: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True)