"""
app_streamlit.py
Simple Streamlit app to load saved model and predict spam or ham.
Run:
    streamlit run app_streamlit.py --server.port 8501
"""

import streamlit as st
import joblib
import os
import email
from io import StringIO

# Helper: load model
MODEL_PATH = "models/spam_clf.joblib"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model not found at {path}. Train the model first with train_spam.py")
        return None
    model = joblib.load(path)
    return model

# Optionally parse .eml files to extract plain text body
def extract_eml_text(file_bytes):
    try:
        msg = email.message_from_bytes(file_bytes)
        parts = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == 'text/plain' and part.get_payload(decode=True):
                    parts.append(part.get_payload(decode=True).decode(errors='ignore'))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                parts.append(payload.decode(errors='ignore'))
        return "\n".join(parts).strip()
    except Exception as e:
        return ""

st.title("Spam Detector (TF-IDF + Naive Bayes)")
st.write("Paste a message or upload a `.txt` / `.eml` file to classify as Spam or Not Spam.")

model = load_model()
if model is None:
    st.stop()

# Text input
user_input = st.text_area("Paste message here", height=200)

# File upload
uploaded_file = st.file_uploader("Or upload file (.txt or .eml)", type=['txt','eml'])
if uploaded_file is not None and uploaded_file.type == 'message/rfc822':
    # .eml
    text_from_eml = extract_eml_text(uploaded_file.read())
    if text_from_eml:
        user_input = text_from_eml

if st.button("Predict"):
    if not user_input:
        st.warning("Please paste text or upload a file.")
    else:
        pred = model.predict([user_input])[0]
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([user_input])[0][1]  # spam probability
        label = "SPAM" if pred == 1 else "NOT SPAM"
        st.markdown(f"### Prediction: **{label}**")
        if prob is not None:
            st.info(f"Spam probability: {prob:.3f}")
        st.write("### Message preview:")
        st.write(user_input)
