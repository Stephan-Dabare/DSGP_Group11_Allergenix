# Allergenix: AI-Driven Allergen Detection App (Group 11)

## Overview
**Allergenix** is an AI-driven application designed to help individuals avoid allergens in everyday products, such as food and cosmetics. By leveraging cutting-edge data science techniques, Allergenix provides users with tools to make safer and more informed decisions.

## Features
- **Ingredient Detection:** Extract ingredient lists from images using:
  - **Convolutional Neural Networks (CNNs):** For detecting and localizing text in images.
  - **OCR Engine:** For accurately extracting text from the detected regions.
- **Allergen Analysis:** Analyze ingredient lists to identify common allergens.
- **Alternative Recommendations:** Suggest safer, alternative products based on analysis results.
- **Data Visualization:** Provide clear and insightful visualizations of the analyzed data to empower users with actionable information.

## Why Allergenix?
Allergenix reduces the risk of allergic reactions by enabling users to:
- Quickly identify potential allergens in products.
- Make informed purchasing decisions with confidence.
- Improve overall quality of life by choosing safer products.

## How It Works
1. **Image Input:** Users capture or upload an image of a product's ingredient label.
2. **Text Extraction:** The app uses CNNs and OCR to extract ingredient information from the image.
3. **Allergen Detection:** The extracted text is analyzed to detect the presence of common allergens.
4. **Recommendations:** Alternative products are suggested if allergens are found.
5. **Data Visualization:** Results are displayed through intuitive charts and graphs for easy interpretation.

## Feature Prototype

Here’s a preview of our system’s feature prototype, which outlines the overall workflow from image upload to allergen detection and alternative product suggestions.
![Feature Prototype](https://terabox.com/s/1SYQUXbpWoq6oQhb1r5MmMA)

## Setup Instructions for Allergenix Application

### Step 1: Download the Allergenix Model Files

1. **Extract the downloaded archive** Allergenix Application Folder 

Due to file size limitations, the required model files are hosted externally. Please follow the steps below to download and set up the application:

2. **Download the model files** from the following cloud storage link:  
   👉 [Download Allergenix Model Files](https://terabox.com/s/1C4xCg0P4C_8O7c7hKm7wOA)  


3. **Copy the extracted folders** into the `model` directory of this repository so that the structure looks like this:

model/
├── 1-Category-Prediction-BERT-Model/
├── 2-Allergen-Detection-BERT-Model/
├── 3-Tokenizer/
└── component_3_models/
└── And other remaining files and folders

### Step 2: Run the Application

Make sure you have all required dependencies installed (check `requirements.txt`. Then run the app using:

python app.py

## Technologies Used
- **Machine Learning Models:** CNN for text detection.
- **Optical Character Recognition (OCR):** For text extraction.
- **Data Visualization:** Tools and libraries for generating user-friendly insights.

## Future Enhancements
- Expand the database of allergen factors.
- Incorporate multi-language support for ingredient detection.
- Add personalized allergen profiles for tailored recommendations.

## About Us

We are a team of second-year undergraduate students pursuing a degree in **Artificial Intelligence and Data Science**. This project is part of our data science group project and aims to apply machine learning and computer vision techniques to real-world problems. Our group project focuses on detecting ingredient labels from packaged food and cosmetic products and analyzing them for potential allergens.

### Team Members and Roles

- **Stephan** – Image-based product detection and text extraction system – Food Items
- **Hirudika** – Image-based product detection and text extraction system – Cosmetic Items
- **Angathan** – Comprehensive Ingredient Analysis and Counselling – Food Items 
- **Uthum** – Comprehensive Ingredient Analysis and Counselling - Cosmetic Items

We worked collaboratively on data collection, preprocessing, model training, and system integration to deliver a complete end-to-end pipeline.

### Our Goal

To build a smart system that:
1. Detects ingredient labels in uploaded images,
2. Extracts and cleans the text,
3. Identifies potential allergens, and
4. Suggests safer alternatives to users with allergies.

We hope our work can contribute to health-conscious decision-making for individuals with dietary and cosmetic sensitivities.

By using **Allergenix**, consumers can reduce the risk of allergic illnesses and lead a healthier life. Explore the app and make informed choices today!
