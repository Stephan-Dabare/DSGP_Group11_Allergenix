import json
import torch
import numpy as np
import pandas as pd
import re
import joblib
from torch.nn.functional import cosine_similarity
from ingredients_analyzer import IngredientsAnalyzer


def normalize_ingredient(ingredient):
    """Normalize ingredient names for consistent comparison"""
    if not isinstance(ingredient, str):
        ingredient = str(ingredient)
    ingredient = ingredient.lower()  # Convert to lowercase
    ingredient = re.sub(r'[-.,/]+', ' ', ingredient)  # Replace separators with space
    ingredient = re.sub(r'\s+', ' ', ingredient).strip()  # Remove extra spaces
    return ingredient


######################################################################################################

def load_product_data(filepath="data/new_product_info3.csv"):
    """
    Load product data from CSV file and process embedding columns.

    Args:
        filepath: Path to CSV file containing product data

    Returns:
        Pandas DataFrame with processed data
    """
    data = pd.read_csv(filepath)

    # Convert JSON string representations back to NumPy arrays
    embedding_columns = ["ingredients_embedding", "concern_chems_embedding", "red_list_embedding", "the_gens_embedding"]
    for col in embedding_columns:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: np.array(json.loads(x)) if not pd.isna(x) else np.zeros(100))

    # Process ingredient lists
    for col in ["concerning_chems_detected", "red_list_chems_detected", "allergens_detected"]:
        if col not in data.columns:
            data[col] = [[] for _ in range(len(data))]
        else:
            # Apply the safe processing function
            data[col] = data[col].apply(safe_process_ingredients)

    return data


def compute_similarity(input_emb, product_emb):
    """
    Compute cosine similarity between two embeddings.
    """
    try:
        # Convert numpy arrays to tensors
        if isinstance(input_emb, np.ndarray):
            input_emb = torch.tensor(input_emb, dtype=torch.float32)
        if isinstance(product_emb, np.ndarray):
            product_emb = torch.tensor(product_emb, dtype=torch.float32)

        # Ensure 1D by flattening if needed
        if len(input_emb.shape) > 1:
            input_emb = input_emb.flatten()
        if len(product_emb.shape) > 1:
            product_emb = product_emb.flatten()

        # Match dimensions if different lengths
        if input_emb.shape[0] != product_emb.shape[0]:
            min_len = min(input_emb.shape[0], product_emb.shape[0])
            input_emb = input_emb[:min_len]
            product_emb = product_emb[:min_len]

        return cosine_similarity(input_emb.unsqueeze(0), product_emb.unsqueeze(0)).item()
    except Exception:
        return 0.0


def safe_process_ingredients(item):
    """
    Safely process ingredient lists avoiding pd.isna() issues.
    """
    # Handle None values
    if item is None:
        return []

    # Check if it's already a list
    if isinstance(item, list):
        # Process each element to handle potential nested lists
        result = []
        for elem in item:
            if isinstance(elem, str):
                # Check if the string looks like a list representation
                if elem.strip().startswith('[') and elem.strip().endswith(']'):
                    try:
                        # Try different parsing methods
                        try:
                            # First try JSON parsing with proper quote replacement
                            parsed = json.loads(elem.replace("'", '"'))
                            if isinstance(parsed, list):
                                result.extend(parsed)
                            else:
                                result.append(parsed)
                        except:
                            # Then try ast.literal_eval
                            import ast
                            try:
                                parsed = ast.literal_eval(elem)
                                if isinstance(parsed, list):
                                    result.extend(parsed)
                                else:
                                    result.append(parsed)
                            except:
                                # If all parsing fails, keep the original string
                                result.append(elem)
                    except:
                        result.append(elem)
                else:
                    # Not a list representation, keep as is
                    result.append(elem)
            elif isinstance(elem, list):
                # Flatten nested lists
                result.extend(elem)
            else:
                # Keep other types as is
                result.append(elem)
        return result

    # Handle string values
    if isinstance(item, str):
        if not item or item == '[]':
            return []

        # Check if it's a JSON-like string representation of a list
        if item.strip().startswith('[') and item.strip().endswith(']'):
            try:
                # Try to parse as JSON with proper quote replacement
                parsed = json.loads(item.replace("'", '"'))
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except:
                # Try with ast.literal_eval if JSON parsing fails
                try:
                    import ast
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, list):
                        return parsed
                    return [parsed]
                except:
                    # If all parsing fails, treat as a single item
                    return [item]

        # Handle comma-separated strings
        if ',' in item:
            return [x.strip() for x in item.split(',')]

        # Single value string
        return [item]

    # For any other type, try to convert to a list or return empty list
    try:
        return list(item)
    except:
        return []


def safe_len(x):
    """Helper function to safely get the length of a list."""
    if isinstance(x, list):
        return len(x)
    return 0


def format_list(ingredients):
    """
    Format a list of ingredients for display.
    """
    if ingredients is None:
        return "None detected"

    # Ensure we have a flat list of strings
    flat_list = []
    if isinstance(ingredients, list):
        for item in ingredients:
            if isinstance(item, list):
                flat_list.extend(str(x) for x in item)
            else:
                flat_list.append(str(item))

        if len(flat_list) > 0:
            # Remove any quotes from strings
            clean_list = [item.strip('"\'') for item in flat_list]
            return ", ".join(clean_list)
        else:
            return "None detected"

    # Handle string input
    elif isinstance(ingredients, str):
        return ingredients if ingredients else "None detected"

    return "None detected"


def has_matching_unsafe_ingredients(product_concern, product_allergens, product_red_list,
                                    concern_chems_include, the_gens_include, red_list_include):
    """
    Check if product contains any of the user's input unsafe ingredients.
    """
    # Using normalize_ingredient imported from ingredients_analyzer
    normalized_product_concern = [normalize_ingredient(str(item)) for item in product_concern]
    normalized_product_allergens = [normalize_ingredient(str(item)) for item in product_allergens]
    normalized_product_red_list = [normalize_ingredient(str(item)) for item in product_red_list]

    # Check if any input concerning chemicals match product's concerning chemicals
    for item in concern_chems_include:
        if item in normalized_product_concern:
            return True

    # Check if any input allergens match product's allergens
    for item in the_gens_include:
        if item in normalized_product_allergens:
            return True

    # Check if any input red list ingredients match product's red list
    for item in red_list_include:
        if item in normalized_product_red_list:
            return True

    return False


def calculate_similarity_score(input_embedding, filtered_data,
                               concern_chems_include, the_gens_include, red_list_include):
    """
    Calculate similarity scores between input embedding and product data.
    """
    # Ensure input_embedding is 1D
    if isinstance(input_embedding, np.ndarray) and len(input_embedding.shape) > 1 and input_embedding.shape[0] == 1:
        input_embedding = input_embedding.flatten()

    # Create a copy to avoid modifying the original dataframe
    scored_data = filtered_data.copy()

    # Calculate similarity for ingredients using embeddings
    scored_data["ingredient_similarity"] = scored_data["ingredients_embedding"].apply(
        lambda x: compute_similarity(input_embedding, x)
    )

    # Calculate penalty scores based on concerning ingredients
    scored_data["concerning_chems_score"] = scored_data["concerning_chems_detected"].apply(
        lambda x: 0.02 * safe_len(x)
    )
    scored_data["red_list_score"] = scored_data["red_list_chems_detected"].apply(
        lambda x: 0.1 * safe_len(x)
    )
    scored_data["allergens_score"] = scored_data["allergens_detected"].apply(
        lambda x: 0.04 * safe_len(x)
    )

    # Calculate final score - using embedding similarity minus penalties
    scored_data["final_score"] = scored_data.apply(
        lambda row: row["ingredient_similarity"] - row["concerning_chems_score"]
                    - row["red_list_score"] - row["allergens_score"],
        axis=1
    )

    # Filter out products with matching unsafe ingredients and low scores
    filtered_products = []
    eliminated_count = 0
    low_score_count = 0  # Track products eliminated due to low scores

    for idx, product in scored_data.iterrows():
        unsafe = has_matching_unsafe_ingredients(
            product.get('concerning_chems_detected', []),
            product.get('allergens_detected', []),
            product.get('red_list_chems_detected', []),
            concern_chems_include, the_gens_include, red_list_include
        )

        # Check for unsafe ingredients AND low score
        if unsafe:
            eliminated_count += 1
        elif product["final_score"] < 0.9:  # Check if score is too low
            low_score_count += 1
        else:
            filtered_products.append(product)

    # Convert filtered products list back to DataFrame for sorting
    if filtered_products:
        safe_products_df = pd.DataFrame(filtered_products)
        return safe_products_df, eliminated_count, low_score_count
    else:
        return pd.DataFrame(), eliminated_count, low_score_count


def predict_category(input_embedding, model_path="data/model/new_random_forest.pkl"):
    """
    Predict the category of a product based on its embedding.
    """
    try:
        # Load the saved model
        classifier = joblib.load(model_path)

        # Reshape input embedding if needed
        if len(input_embedding.shape) == 1:
            input_embedding = input_embedding.reshape(1, -1)

        # Predict the category
        predicted_category = classifier.predict(input_embedding)
        return predicted_category[0]
    except Exception as e:
        print(f"Error predicting category: {e}")
        return "Unknown"


def get_recommendation(input_embedding, primary_category=None, selected_tertiary=None,
                       concern_chems_include=[], the_gens_include=[], red_list_include=[],
                       data=None, model_path="data/model/new_random_forest.pkl"):
    """
    Get product recommendations based on input embedding and filters.
    """
    # Define category mapping
    tertiary_categories_map = {
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

    # Load data if not provided
    if data is None:
        data = load_product_data()

    # Predict category if not provided
    if primary_category is None:
        primary_category = predict_category(input_embedding, model_path)

    # Check if category is valid and has available tertiary categories
    if primary_category == "Other" or primary_category not in tertiary_categories_map:
        return {
            "status": "error",
            "message": "No recommendations available for this category.",
            "primary_category": primary_category,
            "recommendations": []
        }

    # Get available tertiary categories
    available_tertiary = tertiary_categories_map.get(primary_category, [])

    # If no tertiary categories selected, use all
    if not selected_tertiary:
        selected_tertiary = available_tertiary

    # Filter data by primary category and selected tertiary categories
    filtered_data = data[
        (data["primary_category"] == primary_category) &
        (data["tertiary_category"].isin(selected_tertiary))
        ].copy()

    if len(filtered_data) == 0:
        return {
            "status": "error",
            "message": "No products found in these categories. Try different categories.",
            "primary_category": primary_category,
            "selected_tertiary": selected_tertiary,
            "recommendations": []
        }

    # Calculate similarity scores
    safe_products_df, eliminated_count, low_score_count = calculate_similarity_score(
        input_embedding, filtered_data,
        concern_chems_include, the_gens_include, red_list_include
    )

    if len(safe_products_df) == 0:
        return {
            "status": "no_matches",
            "message": "All products were eliminated due to unsafe ingredients or low scores.",
            "eliminated_unsafe": eliminated_count,
            "eliminated_low_score": low_score_count,
            "primary_category": primary_category,
            "selected_tertiary": selected_tertiary,
            "recommendations": []
        }

    # Sort by final score and get top products
    top_products = safe_products_df.sort_values(by="final_score", ascending=False).head(3)

    # Format recommendations
    recommendations = []
    for idx, product in top_products.iterrows():
        recommendation = {
            "name": product['product_name'],
            "brand": product.get('brand_name', 'Unknown'),
            "category": product['tertiary_category'],
            "similarity": float(product['ingredient_similarity']),
            "final_score": float(product['final_score']),
            "concerns": {
                "concerning_chemicals": {
                    "list": format_list(product.get('concerning_chems_detected', [])),
                    "score": float(product['concerning_chems_score'])
                },
                "allergens": {
                    "list": format_list(product.get('allergens_detected', [])),
                    "score": float(product['allergens_score'])
                },
                "red_list": {
                    "list": format_list(product.get('red_list_chems_detected', [])),
                    "score": float(product['red_list_score'])
                }
            }
        }
        recommendations.append(recommendation)

    # Return comprehensive results
    return {
        "status": "success",
        "message": f"Found {len(recommendations)} high-quality safe products.",
        "primary_category": primary_category,
        "selected_tertiary": selected_tertiary,
        "eliminated_unsafe": eliminated_count,
        "eliminated_low_score": low_score_count,
        "total_matches": len(safe_products_df),
        "recommendations": recommendations
    }