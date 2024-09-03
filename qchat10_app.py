import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background-color: #4CAF50; color: white; }
    h1 { color: #4CAF50; }
    .stRadio label { font-size: 18px; color: #333; }
    .result-text { font-size: 20px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Load pre-trained model (replace with your actual model path)
model_path = "qchat10_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# QChat10 Questions
questions = [
    "Does your child take an interest in other children?",
    "Does your child ever use his/her index finger to point, to ask for something?",
    "Does your child ever use his/her index finger to point, to indicate interest in something?",
    "Can your child play properly with small toys?",
    "Does your child ever bring objects over to you to show you something?",
    "Does your child look at you when you call his/her name?",
    "Does your child smile in response to your face or smile?",
    "Does your child imitate you (e.g., you make a face, will your child imitate it)?",
    "Does your child respond to his/her name when you call?",
    "If you point at a toy across the room, does your child look at it?"
]

# Streamlit UI
st.title("QChat10 Autism Screening")

st.write("""
    **Welcome to the QChat10 Autism Screening Tool**.
    Please answer the following questions based on your observations of your child.
""")

# Split layout into two columns
col1, col2 = st.columns(2)
user_responses = []

# Collect user responses using select_slider for better UI
for i, question in enumerate(questions):
    response = col1.select_slider(
        f"{i+1}. {question}",
        options=["No", "Yes"],
        key=f"question_{i}"
    )
    user_responses.append(1 if response == "Yes" else 0)

if col2.button("Submit"):
    # Convert user responses to numpy array
    user_data = np.array(user_responses).reshape(1, -1)

    # Make prediction
    prediction = model.predict(user_data)[0]
    prediction_proba = model.predict_proba(user_data)[0][1]

    # Display results
    st.markdown("---")
    if prediction == 1:
        st.error(f"<div class='result-text'>Higher likelihood of autism detected. Probability: {prediction_proba:.2f}</div>", unsafe_allow_html=True)
        st.write("Please consult with a healthcare professional for a detailed evaluation.")
    else:
        st.success(f"<div class='result-text'>Lower likelihood of autism detected. Probability: {prediction_proba:.2f}</div>", unsafe_allow_html=True)
        st.write("It is still recommended to monitor your child's development and consult with a healthcare professional if you have any concerns.")

# Option to reset the form
if st.button("Reset"):
    st.experimental_rerun()
