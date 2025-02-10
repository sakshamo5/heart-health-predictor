import streamlit as st
import pickle

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    with open("spam_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit UI
st.title("📧 Spam Mail Detection")
st.write("Enter an email below to check if it's spam or not.")

# Input email text
email_text = st.text_area("✉️ Email Content:")

if st.button("🔍 Check Spam Status"):
    if email_text.strip():
        # Transform text using vectorizer
        email_vectorized = vectorizer.transform([email_text])
        prediction = model.predict(email_vectorized)

        # Display result
        if prediction[0] == 1:
            st.error("🚨 This is a SPAM email!")
        else:
            st.success("✅ This is NOT spam.")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
