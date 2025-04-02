from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
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
from PIL import ImageFile, Image
import base64
import io
import json
import matplotlib
from matplotlib import pyplot as plt
from product_recommender import predict_category as predict_category_from_embedding, get_recommendation, load_product_data

# Use non-GUI backend for matplotlib
matplotlib.use('Agg')

# ------------------------------
# Main Component Imports and Initialization
# ------------------------------
import component_one
from component_one import process_image, init_component_one
from component_two import ComponentTwo
from database_manager import UserDatabase

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize the user database and main components
db = UserDatabase()
init_component_one()
component_two = ComponentTwo()

# ------------------------------
# Secondary Component: Global Variables, Model Loading, and Utility Functions
# ------------------------------

# Global variable to store extracted ingredients text (for the secondary component)
extracted_ingredients_text = None

# Load EfficientNetV2L as feature extractor
base_model = EfficientNetV2L(weights='imagenet', include_top=False, pooling='avg')

# Load One-Class SVM and preprocessing models
pca = joblib.load("model/component_3_models/pca_model.pkl")
scaler = joblib.load("model/component_3_models/scaler.pkl")
minmax_scaler = joblib.load("model/component_3_models/minmax_scaler.pkl")
oc_svm = joblib.load("model/component_3_models/ocsvm_model.pkl")

# Load fine-tuned BERT model and tokenizer
model_path_2 = "model/component_3_models/fine_tuned_bert_____"
try:
    model_2 = BertForSequenceClassification.from_pretrained(model_path_2)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_path_2)
except Exception as e:
    print(f"Error loading model: {e}")

# Keywords for text extraction
KEYWORDS = ["key ingredients", "key ingredient", "ingredient", "ingredients", "content", "component", "composition"]

# Import and initialize the ingredients analyzer (assumes ingredients_analyzer.py is available)
from ingredients_analyzer import IngredientsAnalyzer

analyzer = IngredientsAnalyzer()

# Mapping of primary to tertiary categories
TERTIARY_CATEGORIES_MAP = {
    "Skincare": [
        "Anti-Aging", "BB & CC Cream", "Blemish & Acne Treatments", "Body Sunscreen",
        "Exfoliators", "Eye Creams & Treatments", "Eye Masks", "Face Masks",
        "Face Oils", "Face Serums", "Face Sunscreen", "Face Wash & Cleansers",
        "Facial Peels", "For Body", "For Face", "Mists & Essences",
        "Moisturizers", "Night Creams", "Sheet Masks", "Toners"
    ],
    "Makeup": [
        "Blush", "Bronzer", "Cheek Palettes", "Color Correct",
        "Concealer", "Contour", "Eye Palettes", "Eye Primer",
        "Eye Sets", "Eyebrow", "Eyeliner", "Eyeshadow",
        "Face Primer", "Face Sets", "False Eyelashes", "Foundation",
        "Highlighter", "Lip Balm & Treatment", "Lip Gloss", "Lip Liner",
        "Lip Plumper", "Lip Sets", "Lipstick", "Liquid Lipstick",
        "Makeup Removers", "Mascara", "Setting Spray & Powder", "Tinted Moisturizer"
    ],
    "Hair": [
        "Conditioner", "Dry Shampoo", "Hair Masks", "Hair Oil",
        "Hair Primers", "Hair Spray", "Hair Styling Products",
        "Leave-In Conditioner", "Scalp Treatments", "Shampoo"
    ],
    "Fragrance": [
        "Body Mist & Hair Mist", "Cologne", "Perfume", "Rollerballs & Travel Size"
    ]
}


# # Function to preprocess image for classification (used in secondary component)
# def preprocess_image_for_classification(image_path):
#     img = image.load_img(image_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return preprocess_input(img_array)
#
#
# def classify_image(image_input):
#     try:
#         # If image_input is a file path, process it accordingly;
#         # if it is a PIL Image, temporarily save it to process.
#         if isinstance(image_input, str):
#             img_array = preprocess_image_for_classification(image_input)
#         else:
#             # Convert to RGB mode if the image has an alpha channel
#             if image_input.mode == 'RGBA':
#                 image_input = image_input.convert('RGB')
#
#             temp_path = "temp_image.jpg"
#             image_input.save(temp_path)
#             img_array = preprocess_image_for_classification(temp_path)
#             os.remove(temp_path)
#
#         feature = base_model.predict(img_array)
#         feature = np.array(feature, dtype=np.float64)
#         feature_scaled = scaler.transform(feature)
#         feature_pca = pca.transform(feature_scaled)
#         feature_normalized = minmax_scaler.transform(feature_pca)
#         prediction = oc_svm.predict(feature_normalized)
#         return "Normal" if prediction[0] == 1 else "Anomaly"
#     except Exception as e:
#         return f"Error: {str(e)}"


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



# Function to preprocess a PIL image for OCR
def preprocess_image_ocr(pil_image):
    # Convert PIL image (RGB) to BGR for OpenCV processing
    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
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
        line = re.sub(r"^-+", "", line).strip()
        words = line.split()
        while words and (len(words[0]) <= 2 and words[0].isalpha()):
            words.pop(0)
        while words and (len(words[-1]) <= 2 and words[-1].isalpha()):
            words.pop()
        cleaned_line = " ".join(words)
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    return cleaned_lines


# Function to classify an ingredient using the fine-tuned BERT model
def is_cosmetic_ingredient(ingredient, threshold=0.90):
    inputs = tokenizer_2(ingredient, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model_2(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
    cosmetic_prob = probs[0][1]  # Probability for being a cosmetic ingredient (class 1)
    return cosmetic_prob >= threshold


# Function to extract and classify cosmetic ingredients from an image file
def extract_cosmetic_ingredients(image_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        pil_image = Image.open(image_path)
        processed_image = preprocess_image_ocr(pil_image)
        extracted_text = pytesseract.image_to_string(processed_image, config='--oem 3 --psm 6')
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
    except Exception:
        return []


# ------------------------------
# Routes for Main Component
# ------------------------------

@app.route('/')
def home():
    # Main home page
    return render_template('home.html', detected=True)


@app.route('/component_1')
def component_1():
    detected = session.get('detected', False)
    return render_template('component_1.html', detected=detected)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(upload_path)
    ingredients, img_path, detected = process_image(upload_path)
    session['ingredients'] = ingredients
    session['img_path'] = img_path
    session['detected'] = detected
    return render_template('component_1.html',
                           ingredients=ingredients,
                           img_path=img_path,
                           detected=detected)


@app.route('/analyze')
def analyze_allergens():
    ingredients = session.get('ingredients', [])
    if not ingredients:
        return redirect(url_for('home'))
    ingredients_text = ", ".join(ingredients)
    allergens = component_two.detect_allergens(ingredients_text)
    category = component_two.predict_category(ingredients_text)
    alternative = component_two.recommend_alternative(category, allergens, ingredients_text)
    total = len(ingredients_text.split(","))
    allergen_count = sum(
        1 for ingredient in ingredients if any(allergen in ingredient.lower() for allergen in allergens))
    normal_count = total - allergen_count
    print(f"✅ Total Ingredients: {total}, Allergen Count: {allergen_count}, Normal Count: {normal_count}")
    return render_template('component_2.html',
                           ingredients=ingredients,
                           allergens=allergens,
                           category=category,
                           alternative=alternative,
                           allergen_count=allergen_count,
                           normal_count=normal_count,
                           total=total)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if db.add_user(username, email, password):
            return redirect(url_for('login'))
        else:
            return "Username or Email already exists! Try another one."
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if db.verify_user(email, password):
            session['email'] = email
            return redirect(url_for('home'))
        else:
            return "Invalid email or password!"
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully!", "info")
    return redirect(url_for('home'))


# ------------------------------
# Routes for Secondary Component
# ------------------------------

@app.route('/component_3')
def component_3():
    # Entry point for the secondary component.
    # (Link your “Detect Now” button to this route.)
    return render_template('component_3.html')


@app.route('/extract_text', methods=['POST'])
def extract_text():
    global extracted_ingredients_text
    try:
        image_data = request.json['image']
        image_data = image_data.split(',')[1]  # Remove the data URL prefix
        pil_image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Convert to RGB if image has alpha channel
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')

        # Step 1: Classify image
        classification = classify_image(pil_image)
        if classification == "Non-Cosmetic":
            return jsonify({'status': 'success', 'classification': classification, 'text': []})

        # Step 2: Save the image temporarily (since extract_cosmetic_ingredients expects a file path)
        temp_path = "temp_image_for_extraction.jpg"
        pil_image.save(temp_path)
        cosmetic_ingredients = extract_cosmetic_ingredients(temp_path)
        os.remove(temp_path)

        # Rest of the function remains the same...

        if not cosmetic_ingredients:
            return jsonify({'status': 'success', 'classification': "Non-Cosmetic", 'text': []})

        # Step 3: Process and clean ingredients
        processed_ingredients = []
        temp_str = ""
        for line in cosmetic_ingredients:
            line = line.strip()
            if not line.endswith(","):
                temp_str += " " + line
            else:
                temp_str += " " + line
                processed_ingredients.append(temp_str.strip())
                temp_str = ""
        if temp_str:
            processed_ingredients.append(temp_str.strip())

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
        if final_classification == "Cosmetic":
            extracted_ingredients_text = ', '.join(cleaned_ingredients)

        return jsonify({'status': 'success', 'classification': final_classification, 'text': cleaned_ingredients})
    except Exception as e:
        print(f"Error in extract_text: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/analysis')
def analysis():
    """Process ingredients and show analysis results."""
    global extracted_ingredients_text
    if not extracted_ingredients_text:
        return "No ingredients extracted. Please upload an image first.", 400

    # Analyze ingredients using the dynamic input
    analysis_result = analyzer.analyze_ingredients(extracted_ingredients_text)

    # Generate visualization
    fig = analyzer.visualize_ingredients(analysis_result=analysis_result)
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png', bbox_inches='tight')
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
    plt.close(fig)

    # Predict primary category using the embedding-based approach
    try:
        primary_category = predict_category_from_embedding(analysis_result['embedding'])
    except Exception as e:
        print(f"Error predicting category: {e}")
        primary_category = "Unknown"

    results = {
        'analysis': {
            'chart_image': img_base64,
            'good_count': analysis_result['counts']['good'],
            'high_risk_count': analysis_result['counts']['high_risk'],
            'average_risk_count': analysis_result['counts']['average_risk'],
            'low_risk_count': analysis_result['counts']['low_risk'],
            'total_count': analysis_result['counts']['total'],
            'ingredients': analysis_result['all_ingredients'],
            'red_list': analysis_result['red_list_ingredients'],
            'allergens': analysis_result['allergen_ingredients'],
            'concerns': analysis_result['concern_ingredients'],
            'embedding': json.dumps(analysis_result['embedding'].tolist()),
            'primary_category': primary_category
        }
    }

    return render_template('component_4_analysis.html', results=results)

@app.route('/recommendation', methods=['POST'])
def recommendation():
    """Show recommendation form with tertiary categories."""
    try:
        embedding = request.form.get('embedding')
        red_list = request.form.get('red_list')
        allergens = request.form.get('allergens')
        concerns = request.form.get('concerns')
        primary_category = request.form.get('primary_category')

        try:
            embedding_data = json.loads(embedding)
        except:
            embedding_data = []
        try:
            red_list_data = json.loads(red_list)
        except:
            red_list_data = []
        try:
            allergens_data = json.loads(allergens)
        except:
            allergens_data = []
        try:
            concerns_data = json.loads(concerns)
        except:
            concerns_data = []

        embedding_array = np.array(embedding_data)
        if not primary_category or primary_category not in TERTIARY_CATEGORIES_MAP:
            primary_category = predict_category_from_embedding(embedding_array)

        tertiary_categories = TERTIARY_CATEGORIES_MAP.get(primary_category, [])
        if primary_category == "Other" or primary_category == "Unknown":
            return render_template('component_4_recommendation.html',
                                   embedding=json.dumps(embedding_data),
                                   red_list=json.dumps(red_list_data),
                                   allergens=json.dumps(allergens_data),
                                   concerns=json.dumps(concerns_data),
                                   primary_category=primary_category,
                                   tertiary_categories=[],
                                   recommendations=[],
                                   no_products_error=False)

        return render_template('component_4_recommendation.html',
                               embedding=json.dumps(embedding_data),
                               red_list=json.dumps(red_list_data),
                               allergens=json.dumps(allergens_data),
                               concerns=json.dumps(concerns_data),
                               primary_category=primary_category,
                               tertiary_categories=tertiary_categories,
                               recommendations=[],
                               no_products_error=False)
    except Exception as e:
        print(f"Error in recommendation route: {str(e)}")
        return f"An error occurred: {str(e)}", 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Generate and display product recommendations."""
    try:
        embedding = json.loads(request.form.get('embedding'))
        red_list = json.loads(request.form.get('red_list'))
        allergens = json.loads(request.form.get('allergens'))
        concerns = json.loads(request.form.get('concerns'))
        primary_category = request.form.get('primary_category')
        selected_tertiary = request.form.getlist('tertiary_categories')

        embedding_array = np.array(embedding)
        product_data = load_product_data()
        recommendation_result = get_recommendation(
            input_embedding=embedding_array,
            primary_category=primary_category,
            selected_tertiary=selected_tertiary,
            concern_chems_include=concerns,
            the_gens_include=allergens,
            red_list_include=red_list,
            data=product_data
        )

        no_products_error = False
        if recommendation_result['status'] == 'error' or not recommendation_result['recommendations']:
            no_products_error = True

        return render_template('component_4_recommendation.html',
                               embedding=json.dumps(embedding),
                               red_list=json.dumps(red_list),
                               allergens=json.dumps(allergens),
                               concerns=json.dumps(concerns),
                               primary_category=primary_category,
                               tertiary_categories=TERTIARY_CATEGORIES_MAP.get(primary_category, []),
                               recommendations=recommendation_result['recommendations'],
                               no_products_error=no_products_error)
    except Exception as e:
        print(f"Error in get_recommendations route: {str(e)}")
        return f"An error occurred: {str(e)}", 500


# ------------------------------
# Run the App
# ------------------------------
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
