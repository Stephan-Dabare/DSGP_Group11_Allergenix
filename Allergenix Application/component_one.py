import os
import re
import cv2
import pytesseract
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from spellchecker import SpellChecker
from fuzzywuzzy import process
import numpy as np
import joblib
from gensim.models import KeyedVectors


def init_component_one():
    global yolo_model, dataset_path, word2vec_model, classifier
    print("⚙️ Initializing component one models...")

    try:
        print("🔍 Loading YOLO model...")
        yolo_model = YOLO('model/best.pt')
        if not os.path.exists('model/best.pt'):
            raise FileNotFoundError("YOLO model file not found")

        print("📖 Loading Word2Vec model...")
        word2vec_model = KeyedVectors.load("model/word2vec_model.kv", mmap='r')

        print("🤖 Loading classifier...")
        classifier = joblib.load('model/ingredient_classifier_model_w2v.pkl')

        dataset_path = os.path.normpath("data/4-Filtered-OpenFoodFacts.csv")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        print("✅ Component one initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        raise


def get_word2vec_vector(text):
    words = text.split()
    vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.sum(vectors, axis=0) / len(vectors) if vectors else np.zeros(100)


def process_image(image_path, match_threshold=80):
    try:
        print(f"\n🔍 Processing image: {os.path.basename(image_path)}")

        # YOLO Detection
        print("=== YOLO Detection ===")
        results = yolo_model.predict(source=image_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            print("⚠️ No ingredients detected")
            return None, "No ingredient section detected", False
        print(f"📦 Detected {len(boxes)} bounding boxes")

        # Bounding Box Processing
        print("=== Bounding Box ===")
        x1, y1, x2, y2 = map(int, boxes[0])
        img = cv2.imread(image_path)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        output_path = "static/processed/detected_ingredients.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"💾 Saved bounding box to {output_path}")

        # OCR Processing
        print("=== OCR Processing ===")
        cropped_img = Image.fromarray(img[y1:y2, x1:x2])
        text = pytesseract.image_to_string(cropped_img).lower()
        print(f"📝 Raw OCR text: {text[:200]}...")

        # Text Processing
        print("=== Text Correction ===")
        raw_tokens = re.findall(r'\b[\w-]+\b', text)
        spell = SpellChecker()
        corrected_tokens = [spell.correction(token) or token for token in raw_tokens if token.strip()]
        print(f"✏️ Corrected tokens: {corrected_tokens}")

        # Classification
        print("=== Classification ===")
        food_ingredients = []
        for token in corrected_tokens:
            vector = get_word2vec_vector(token)
            if vector is not None:
                prediction = classifier.predict(vector.reshape(1, -1))
                if prediction[0] == 1:
                    food_ingredients.append(token)
        print(f"🍎 Food ingredients: {food_ingredients}")

        # Fuzzy Matching
        print("=== Fuzzy Matching ===")
        dataset = pd.read_csv(dataset_path)
        if 'ingredients_text' not in dataset.columns:
            raise KeyError("Dataset missing 'ingredients_text' column")

        valid_ingredients = dataset['ingredients_text'].str.lower().tolist()
        final_ingredients = []

        for ingredient in food_ingredients:
            match, score = process.extractOne(ingredient, valid_ingredients)
            if score >= match_threshold:
                final_ingredients.append(match)

        final_ingredients = list(dict.fromkeys(final_ingredients))
        print(f"🎯 Final ingredients: {final_ingredients}")

        # Return relative path for web display
        relative_output_path = output_path.replace("static/", "")
        return final_ingredients, relative_output_path, True

    except Exception as e:
        print(f"🔥 Processing error: {str(e)}")
        return None, f"Error: {str(e)}", False