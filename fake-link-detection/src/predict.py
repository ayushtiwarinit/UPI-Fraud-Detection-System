# Prediction logic
import pickle
import os
import warnings
from src.preprocess import clean_message, analyze_url_structure

warnings.filterwarnings("ignore", category=UserWarning)

def check_message(user_input):
    vec_path = os.path.join('models', 'vectorizer.pkl')
    mod_path = os.path.join('models', 'model.pkl')
    
    if not os.path.exists(vec_path) or not os.path.exists(mod_path):
        return "ERROR: Models not found. Run train.py first."

    with open(vec_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(mod_path, 'rb') as f:
        model = pickle.load(f)
    
    url_risk = analyze_url_structure(user_input)
    
    cleaned = clean_message(user_input)
    
    if not cleaned.strip():
        return "FRAUD" if url_risk == 1 else "SAFE"
        
    vector = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    
    if prediction == 1 or url_risk == 1:
        return "FRAUD"
    
    return "SAFE"
