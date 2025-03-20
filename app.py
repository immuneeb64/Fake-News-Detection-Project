import numpy as np
import joblib
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load models (without "models/" folder)
nb_model = joblib.load("naive_bayes_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
lstm_model = load_model("lstm_model.h5")

# Load vectorizer and tokenizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
tokenizer = joblib.load("tokenizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter a news article to check:")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Convert text to TF-IDF vector
        user_text_tfidf = vectorizer.transform([user_input])  

        # Convert text to LSTM input (tokenization + padding)
        lstm_input = tokenizer.texts_to_sequences([user_input])
        lstm_input = pad_sequences(lstm_input, maxlen=500)

        # Predictions
        nb_prediction = nb_model.predict(user_text_tfidf)[0]  # 0 = Fake, 1 = Real
        rf_prediction = rf_model.predict(user_text_tfidf)[0]  
        lstm_prediction = (lstm_model.predict(lstm_input) > 0.5).astype("int32")[0][0]  

        # 🏆 Final Decision: If any model predicts "Real", classify as "Real"
        final_prediction = "Real" if (nb_prediction == 1 or rf_prediction == 1 or lstm_prediction == 1) else "Fake"

        # Display final decision only
        st.markdown(f"## {final_prediction} News")
