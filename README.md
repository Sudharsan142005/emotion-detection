# Emotion Detection from Text 🧠

This is a simple web application that detects emotions from user input using a pre-trained BERT model. It uses [Hugging Face Transformers](https://huggingface.co/) for the model and [Streamlit](https://streamlit.io/) for the user interface.

## 🚀 Features

- Accepts a sentence input from the user.
- Classifies the emotion using the `nateraw/bert-base-uncased-emotion` model.
- Displays the detected emotion and the model's confidence.

## 📦 Requirements

Make sure you have Python 3.7 or later installed. Install required packages with:

```bash
pip install -r requirements.txt

🏃‍♂️ Running the App
To run the Streamlit app, use the following command:

streamlit run app.py
📚 Model Info
Model used: nateraw/bert-base-uncased-emotion

Framework: Transformers


