from transformers import pipeline
import streamlit as st

# Title of the web app
st.title("Emotion Detection from Text 🧠")

# Load the emotion classification model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

classifier = load_model()

# Text input box
user_input = st.text_input("Enter a sentence to detect emotion:")

# Show results
if user_input:
    result = classifier(user_input)[0]
    st.markdown(f"**Emotion:** {result['label']}")
    st.markdown(f"**Confidence:** {round(result['score']*100, 2)}%")
