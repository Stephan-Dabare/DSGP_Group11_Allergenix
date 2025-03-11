from flask import Flask, render_template, request, redirect, url_for, jsonify
import base64
import io
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend
from matplotlib import pyplot as plt
import json
import numpy as np

from ingredients_analyzer import IngredientsAnalyzer
from product_recommender import predict_category, get_recommendation, load_product_data

app = Flask(__name__)

# Initialize the analyzer
analyzer = IngredientsAnalyzer()

# Sample ingredients (in a real implementation, you would get this from your database or another source)
SAMPLE_INGREDIENTS = r"Aqua/Water/Eau, Cyclopentasiloxane, Propylene Glycol, Cyclohexasiloxane, Parfum/Fragrance, Diethylhexyl Sebacate, Hydroxyethyl Acrylate/Sodium Acryloyldimethyl Taurate Copolymer, Glycerin, Caprylic/Capric Triglyceride, Squalane, Phenoxyethanol, Magnesium Aluminum Silicate, Peg-100 Stearate, Titanium Dioxide, Polysorbate 60, Methylparaben, Xanthan Gum, Aloe Barbadensis Leaf Juice, Bisabolol, Disodium Edta, Aluminum Starch Octenylsuccinate, Propylparaben, Limonene, Linalool, Triethanolamine, Ethylparaben, Coumarin, Hexyl Cinnamal, Farnesol, Eugenol, Citronellol, Allantoin, Geraniol, Alpha-Isomethyl Ionone, Galactoarabinan, Panthenol, Polysorbate 20, Citral, Potassium Sorbate, Ascorbic Acid, Citrus Limon (Lemon) Peel , Tocopherol, Citric Acid, Sodium Benzoate, Bht, Cymbopogon Schoenanthus , Retinyl Palmitate."

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


@app.route('/')
def analysis():
    """Process ingredients and show analysis results."""
    # Get ingredients from your backend (here we use the sample)
    ingredients_str = SAMPLE_INGREDIENTS  # In production, replace with your data source

    # Analyze ingredients
    analysis_result = analyzer.analyze_ingredients(ingredients_str)

    # Generate visualization
    fig = analyzer.visualize_ingredients(analysis_result=analysis_result)

    # Convert plot to base64 string for embedding in HTML
    img_data = io.BytesIO()
    fig.savefig(img_data, format='png', bbox_inches='tight')
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
    plt.close(fig)  # Close the figure to free memory

    # Predict primary category
    try:
        primary_category = predict_category(analysis_result['embedding'])
    except Exception as e:
        print(f"Error predicting category: {e}")
        primary_category = "Unknown"

    # Prepare results for the template
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

    return render_template('analysis.html', results=results)


@app.route('/recommendation', methods=['POST'])
def recommendation():
    """Show recommendation form with tertiary categories."""
    try:
        # Get data passed from analysis page
        embedding = request.form.get('embedding')
        red_list = request.form.get('red_list')
        allergens = request.form.get('allergens')
        concerns = request.form.get('concerns')
        primary_category = request.form.get('primary_category')

        # Parse JSON data safely
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

        # Convert embedding back to numpy array
        embedding_array = np.array(embedding_data)

        # If primary category wasn't passed or is invalid, try to predict it
        if not primary_category or primary_category not in TERTIARY_CATEGORIES_MAP:
            primary_category = predict_category(embedding_array)

        # Get tertiary categories for the primary category
        tertiary_categories = TERTIARY_CATEGORIES_MAP.get(primary_category, [])

        return render_template('recommendation.html',
                               embedding=json.dumps(embedding_data),
                               red_list=json.dumps(red_list_data),
                               allergens=json.dumps(allergens_data),
                               concerns=json.dumps(concerns_data),
                               primary_category=primary_category,
                               tertiary_categories=tertiary_categories,
                               recommendations=[])
    except Exception as e:
        # For debugging
        print(f"Error in recommendation route: {str(e)}")
        return f"An error occurred: {str(e)}", 500


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Generate and display product recommendations."""
    try:
        # Get form data
        embedding = json.loads(request.form.get('embedding'))
        red_list = json.loads(request.form.get('red_list'))
        allergens = json.loads(request.form.get('allergens'))
        concerns = json.loads(request.form.get('concerns'))
        primary_category = request.form.get('primary_category')
        selected_tertiary = request.form.getlist('tertiary_categories')

        # Convert embedding back to numpy array
        embedding_array = np.array(embedding)

        # Load product data
        product_data = load_product_data()

        # Get recommendations using the function from product_recommender
        recommendation_result = get_recommendation(
            input_embedding=embedding_array,
            primary_category=primary_category,
            selected_tertiary=selected_tertiary,
            concern_chems_include=concerns,
            the_gens_include=allergens,
            red_list_include=red_list,
            data=product_data
        )

        return render_template('recommendation.html',
                               embedding=json.dumps(embedding),
                               red_list=json.dumps(red_list),
                               allergens=json.dumps(allergens),
                               concerns=json.dumps(concerns),
                               primary_category=primary_category,
                               tertiary_categories=TERTIARY_CATEGORIES_MAP.get(primary_category, []),
                               recommendations=recommendation_result['recommendations'])
    except Exception as e:
        # For debugging
        print(f"Error in get_recommendations route: {str(e)}")
        return f"An error occurred: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)