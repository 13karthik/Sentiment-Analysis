
import streamlit as st
import joblib

lr_model = joblib.load("logistic_model.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")
rf_model =joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("📊 Sentiment Analysis on Reviews")

review_text = st.text_area("📝 Enter your review text:")

model_choice = st.selectbox("Choose model:", ["Logistic Regression", "Naive Bayes", "Random Forest"])

if st.button("🔍 Predict"):
    if not review_text.strip():
        st.warning("Please enter review text.")
    else:
        vect_input = vectorizer.transform([review_text])
        model = lr_model if model_choice == "Logistic Regression" else nb_model if model_choice == "Naive Bayes" else rf_model
        prediction = model.predict(vect_input)[0]
        sentiment = label_encoder.inverse_transform([prediction])[0]
        st.success(f"Predicted Sentiment: {sentiment}")
