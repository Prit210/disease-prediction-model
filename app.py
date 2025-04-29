from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import re
import json
from langdetect import detect, LangDetectException
from translate import Translator

app = Flask(__name__)
app.debug = True

# === Load model and preprocessing ===
with open('static/final_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('static/imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

with open('static/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('static/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('static/features.json', 'r') as f:
    feature_list = json.load(f)

# === Symptom-to-index map ===
symptom_to_index = {symptom: idx for idx, symptom in enumerate(feature_list)}

# Translate user input to English before symptom extraction
from deep_translator import GoogleTranslator

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print("Translation to English failed:", e)
        return text  # fallback to original

# === Text cleaner ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# === Extract symptoms from input text ===
def extract_symptoms(text):
    extracted = set()
    for symptom in symptom_to_index:
        if re.search(r'\b' + re.escape(symptom.lower()) + r'\b', text):
            extracted.add(symptom)
    return extracted

# Function to translate text
def translate_text_simple(text, target_language):
    # Initialize the translator with the target language
    translator = Translator(to_lang=target_language)
    return translator.translate(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['user_text']

    # # Detect language
    # try:
    #     detected_language = detect(user_text)
    # except LangDetectException:
    #     detected_language = "en"  # Fallback
    
    # user_text=translate_text_simple(user_text, 'en')
    # cleaned = clean_text(user_text)
    # extracted = extract_symptoms(cleaned)

    # Detect language and translate input to English
    detected_language = detect(user_text)
    user_text_en = translate_to_english(user_text)

    # Clean and extract symptoms from English-translated text
    cleaned = clean_text(user_text_en)
    extracted = extract_symptoms(cleaned)

    # === Build binary symptom vector ===
    x = np.zeros(len(feature_list), dtype='float32')
    for sym in extracted:
        x[symptom_to_index[sym]] = 1
    x = x.reshape(1, -1)

    # === Apply preprocessing: imputer → scaler → PCA ===
    x = imputer.transform(x)
    x = scaler.transform(x)
    x = pca.transform(x)

    # === Make predictions ===
    probs = model.predict_proba(x)[0]
    top_idx = np.argsort(probs)[-3:][::-1]
    top = [(model.classes_[i], probs[i]) for i in top_idx]

    # Translate the disease names and probabilities
    translated_details = []
    for d, p in top:
        try:
            translated_name = translate_text_simple(d, detected_language)
            probability_text = translate_text_simple(f"Probability: {p*100:.2f}%", detected_language)
        except Exception as e:
            print(f"Error during translation: {e}")
            translated_name = d
            probability_text = f"Probability: {p*100:.2f}%"
        translated_details.append({
            "name": translated_name,
            "probability": probability_text
        })

    return render_template('predicted.html',
                           disease_details=translated_details,
                           user_text=user_text,
                           language=detected_language)


#     # === Predict single top disease ===
#     predicted_disease = model.predict(x)[0]

# # === Translate disease name to user's language ===
#     from deep_translator import GoogleTranslator

#     try:
#         translated_name = GoogleTranslator(source='en', target=detected_language).translate(predicted_disease)
#     except Exception as e:
#         print("Translation failed:", e)
#         translated_name = predicted_disease

# # === Send to template ===
#     translated_details = {
#        "name": translated_name,
#        "probability": "Top Match"
# }   

#     return render_template('predicted.html',
#                        disease_details=translated_details,
#                        user_text=user_text,
#                        language=detected_language)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
