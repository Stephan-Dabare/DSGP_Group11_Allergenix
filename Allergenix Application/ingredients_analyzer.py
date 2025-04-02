import json
import torch
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec


class IngredientsAnalyzer:
    def __init__(self, model_path="model/component_4_models/ingredient_word2vec.model"):
        """Initialize the analyzer with ingredient lists and word embeddings model."""
        self.word2vec_model = self.load_ingredient_model(model_path)
        self.concern_chems, self.red_list, self.the_gens = self.load_ingredient_lists()

    def load_ingredient_model(self, model_path):
        """Load the trained Word2Vec model."""
        try:
            return Word2Vec.load(model_path)
        except FileNotFoundError:
            print(f"Warning: Model not found at {model_path}.")
            return None

    def load_ingredient_lists(self):
        """Load concern chemicals, red list, and allergens lists."""
        concern_chems = []
        red_list = []
        the_gens = []  # allergens

        try:
            with open("data/concern_chems.txt") as f:
                concern_chems = [self.normalize_ingredient(line.strip()) for line in f]
        except FileNotFoundError:
            print("Warning: concern_chems.txt not found. Using empty list.")

        try:
            with open("data/red_list.txt") as f:
                red_list = [self.normalize_ingredient(line.strip()) for line in f]
        except FileNotFoundError:
            print("Warning: red_list.txt not found. Using empty list.")

        try:
            with open("data/the_gens.txt") as f:
                the_gens = [self.normalize_ingredient(line.strip()) for line in f]
        except FileNotFoundError:
            print("Warning: the_gens.txt not found. Using empty list.")

        return concern_chems, red_list, the_gens

    def get_ingredients(self, ingredients_str):
        """
        Extract and normalize ingredients from a string.

        Args:
            ingredients_str: String containing ingredient list

        Returns:
            List of normalized ingredient strings
        """
        return self.split_and_normalize_ingredients([ingredients_str])

    def normalize_ingredient(self, ingredient):
        """
        Normalize a single ingredient name.

        Args:
            ingredient: An ingredient string

        Returns:
            Normalized ingredient string
        """
        if not isinstance(ingredient, str):
            ingredient = str(ingredient)
        ingredient = ingredient.lower()  # Convert to lowercase
        ingredient = re.sub(r'[-.,/]+', ' ', ingredient)  # Replace separators with space
        ingredient = re.sub(r'\s+', ' ', ingredient).strip()  # Remove extra spaces
        return ingredient

    def split_and_normalize_ingredients(self, ingredients_list):
        """
        Split and normalize a list of ingredient strings.

        Args:
            ingredients_list: List of strings containing ingredients

        Returns:
            List of normalized ingredient strings
        """
        all_ingredients = []
        if isinstance(ingredients_list, str):
            ingredients_list = [ingredients_list]

        for ingredients_str in ingredients_list:
            if not isinstance(ingredients_str, str):
                continue
            # Split by commas while preserving items with parentheses
            ingredients = re.split(r',\s*(?![^()]*\))', ingredients_str)
            # Normalize each ingredient
            normalized_ingredients = [self.normalize_ingredient(ingredient) for ingredient in ingredients]
            all_ingredients.extend(normalized_ingredients)
        return all_ingredients

    def get_sentence_vector(self, sentence):
        """
        Create a vector representation of a sentence using the word2vec model.

        Args:
            sentence: List of words

        Returns:
            numpy array of the sentence embedding
        """
        vectors = [self.word2vec_model.wv[word] for word in sentence if word in self.word2vec_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)  # Take the average of word vectors
        else:
            return np.zeros(self.word2vec_model.vector_size)

    def analyze_ingredients(self, ingredients_str):
        """
        Analyze a list of ingredients for safety and concerns.

        Args:
            ingredients_str: String containing ingredient list

        Returns:
            Dictionary with analysis results
        """
        # Get normalized ingredients
        normalized_ingredients = self.get_ingredients(ingredients_str)

        # Generate embedding
        tokens = simple_preprocess(ingredients_str)
        embedding = self.get_sentence_vector(tokens).reshape(1, -1)

        # Find matches with concern lists
        red_list_include = []
        the_gens_include = []
        concern_chems_include = []

        # Check against red list (high risk)
        for i in normalized_ingredients:
            for j in self.red_list:
                if i == j and j not in red_list_include:
                    red_list_include.append(j)

        # Check against allergens (average risk)
        for i in normalized_ingredients:
            for j in self.the_gens:
                if i == j and j not in red_list_include and j not in the_gens_include:
                    the_gens_include.append(j)

        # Check against concern chemicals (low risk)
        for i in normalized_ingredients:
            for j in self.concern_chems:
                if (i == j and j not in red_list_include and
                        j not in the_gens_include and j not in concern_chems_include):
                    concern_chems_include.append(j)

        # Count safe ingredients
        good_ingredients = [ing for ing in normalized_ingredients
                            if ing not in red_list_include and
                            ing not in the_gens_include and
                            ing not in concern_chems_include]

        # Create analysis result
        result = {
            "all_ingredients": normalized_ingredients,
            "good_ingredients": good_ingredients,
            "red_list_ingredients": red_list_include,
            "allergen_ingredients": the_gens_include,
            "concern_ingredients": concern_chems_include,
            "embedding": embedding,
            "counts": {
                "good": len(good_ingredients),
                "high_risk": len(red_list_include),
                "average_risk": len(the_gens_include),
                "low_risk": len(concern_chems_include),
                "total": len(normalized_ingredients)
            }
        }

        return result

    def visualize_ingredients(self, ingredients_str=None, analysis_result=None):
        """
        Visualize ingredient analysis with a donut chart.

        Args:
            ingredients_str: String containing ingredient list (optional if analysis_result provided)
            analysis_result: Pre-computed analysis result (optional if ingredients_str provided)

        Returns:
            Matplotlib figure
        """
        if analysis_result is None and ingredients_str is not None:
            analysis_result = self.analyze_ingredients(ingredients_str)
        elif analysis_result is None and ingredients_str is None:
            raise ValueError("Either ingredients_str or analysis_result must be provided")

        # Extract counts
        good_count = analysis_result["counts"]["good"]
        high_risk_count = analysis_result["counts"]["high_risk"]
        average_risk_count = analysis_result["counts"]["average_risk"]
        low_risk_count = analysis_result["counts"]["low_risk"]

        # Filter out zero-value categories
        chart_data = []
        labels = []
        colors = []

        if good_count > 0:
            chart_data.append(good_count)
            labels.append('Good')
            colors.append('green')
        if high_risk_count > 0:
            chart_data.append(high_risk_count)
            labels.append('High Risk')
            colors.append('red')
        if average_risk_count > 0:
            chart_data.append(average_risk_count)
            labels.append('Average Risk')
            colors.append('orange')
        if low_risk_count > 0:
            chart_data.append(low_risk_count)
            labels.append('Low Risk')
            colors.append('yellow')

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        # Donut chart parameters
        wedgeprops = {'width': 0.3, 'edgecolor': 'black'}  # Adjust width for hole size
        center_circle = Circle((0, 0), 0.4, color='white')  # Circle in the center
        ax.add_artist(center_circle)

        # Create the pie chart (with the "donut" hole)
        if chart_data:
            wedges, texts, autotexts = ax.pie(
                chart_data,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                labels=labels,
                wedgeprops=wedgeprops,
                textprops={'color': 'black'}
            )

            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')

            # Add title inside the donut hole
            ax.text(
                0, 0,
                "Here's the rating of each\ningredient in your skincare\nproduct!",
                ha='center',
                va='center',
                fontsize=12,
                color='black'
            )

        return fig



