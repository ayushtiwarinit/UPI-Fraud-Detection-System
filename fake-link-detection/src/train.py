import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocess import clean_message

def train_and_save_models():
    print("🚀 Starting Professional Training Pipeline...")

    if not os.path.exists('models'):
        os.makedirs('models')

    try:
        print("📊 Loading dataset...")
        df = pd.read_csv('data/dataset.csv')
        print(f"   -> Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        print("❌ Error: data/dataset.csv not found. Please run your data generation script first.")
        return

    print("🧹 Applying NLP text cleaning (this may take a moment)...")
    df['cleaned'] = df['text'].apply(clean_message)

    print("🔢 Converting text to TF-IDF vectors...")
    tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['cleaned']).toarray()
    y = df['label']

    print("🔀 Splitting data into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("🧠 Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n📈 Evaluating Model Performance on unseen test data...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   -> 🎯 Overall Accuracy: {accuracy * 100:.2f}%")
    
    print("\n   -> 📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Fraud (1)']))
    
    print("   -> 🧮 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n💾 Saving trained models to disk...")
    pickle.dump(tfidf, open('models/vectorizer.pkl', 'wb'))
    pickle.dump(model, open('models/model.pkl', 'wb'))

    print("✅ Pipeline Complete! Models are ready for production in the 'models/' folder.")

if __name__ == "__main__":
    train_and_save_models()
