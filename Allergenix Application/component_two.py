import re
import torch
import pickle
import pandas as pd
from transformers import pipeline, AutoTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ComponentTwo:
    def __init__(self):
        print("\n⚙️ Initializing component two...")
        try:
            # NER Pipeline
            print("🔍 Loading NER model...")
            self.ner_pipeline = pipeline(
                "token-classification",
                model="sgarbi/bert-fda-nutrition-ner",
                aggregation_strategy="simple"
            )

            # Allergen Set
            self.common_allergens = {
                "peanut", "peanuts", "milk", "dairy", "almond", "almonds",
                "wheat", "soy", "egg", "eggs", "cocoa", "chocolate",
                "hazelnut", "hazelnuts", "cashew", "cashews", "walnut", "walnuts"
            }

            # BERT Model
            print("🤖 Loading BERT model...")
            model_path = "model/1-Category-Prediction-BERT-Model"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.classifier_model = BertForSequenceClassification.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.classifier_model.to(self.device)

            # Label Encoder
            print("🏷️ Loading label encoder...")
            with open("model/label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)

            # Dataset
            print("📚 Loading dataset...")
            self.df_filtered = pd.read_csv("data/4-Filtered-OpenFoodFacts.csv")
            print("✅ Component two initialized successfully")

        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            raise

    def detect_allergens(self, ingredients_text):
        print("\n🔍 Detecting allergens...")
        try:
            detected = set()
            ner_results = self.ner_pipeline(ingredients_text)
            for item in ner_results:
                token = item['word'].replace("##", "").lower().strip()
                if token in self.common_allergens:
                    detected.add(token)

            # Fallback check
            tokens = set(re.findall(r'\b\w+\b', ingredients_text.lower()))
            detected.update(tokens.intersection(self.common_allergens))

            print(f"⚠️ Detected allergens: {list(detected)}")
            return list(detected)
        except Exception as e:
            print(f"🔥 Allergen detection failed: {str(e)}")
            return []

    def predict_category(self, text):
        print("\n🔮 Predicting category...")
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.classifier_model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)
            confidence, predicted_label_idx = torch.max(probs, 1)

            category = "Other"
            if confidence.item() >= 0.5:
                category = self.label_encoder.inverse_transform([predicted_label_idx.item()])[0]

            print(f"📌 Predicted category: {category}")
            return category
        except Exception as e:
            print(f"🔥 Category prediction failed: {str(e)}")
            return "Other"

    def recommend_alternative(self, category, allergens, ingredients):
        print("\n💡 Recommending alternatives...")
        try:
            allergen_synonyms = {
                "peanut": ["peanut", "peanuts"],
                "almond": ["almond", "almonds"],
                "milk": ["milk", "dairy", "cream", "cheese", "butter", "yogurt"],
                "cocoa": ["cocoa", "chocolate"],
                "hazelnut": ["hazelnut", "hazelnuts"],
                "cashew": ["cashew", "cashews"],
                "walnut": ["walnut", "walnuts"],
                "soy": ["soy", "soybean", "soya"]
            }

            expanded_allergens = {syn for allergen in allergens for syn in allergen_synonyms.get(allergen, [allergen])}
            print(f"🔍 Expanded allergens: {expanded_allergens}")

            # Filter candidates
            if category.lower() == "other":
                candidate_df = self.df_filtered
            else:
                candidate_df = self.df_filtered[self.df_filtered['category_label'].astype(str).str.lower() == category.lower()]

            # Safety check
            def is_safe(text):
                return expanded_allergens.isdisjoint(set(re.findall(r'\b\w+\b', text.lower())))

            safe_products = candidate_df.copy()[candidate_df["ingredients_text"].fillna("").apply(is_safe)]

            if safe_products.empty:
                safe_products = self.df_filtered.copy()[self.df_filtered["ingredients_text"].fillna("").apply(is_safe)]
                if safe_products.empty:
                    print("⚠️ No alternatives found")
                    return "No allergen-free alternatives found."

            # Similarity calculation
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(safe_products["ingredients_text"])
            input_vec = vectorizer.transform([ingredients])

            safe_products = safe_products.copy()
            safe_products["similarity"] = cosine_similarity(input_vec, tfidf_matrix).flatten()

            best_match = safe_products.sort_values("similarity", ascending=False).iloc[0]['product_name']
            print(f"🎯 Best alternative: {best_match}")
            return best_match

        except Exception as e:
            print(f"🔥 Recommendation failed: {str(e)}")
            return "Error finding alternatives"