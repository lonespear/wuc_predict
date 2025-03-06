import streamlit as st
from model_loader import predict_discrepancy  # Ensure you have model_loader.py with the function above

# Page configuration
st.set_page_config(page_title="KC-135 Work User Code Predictor", page_icon="✈️", layout="centered")

# Custom styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f4f4f4;
        }
        .stTextArea textarea {
            background-color: #ffffff;
            border: 2px solid #2E4053;
            border-radius: 5px;
            font-size: 16px;
            color: #2E4053;
        }
        .stTextArea label {
            font-size: 18px;
            font-weight: bold;
            color: #2E4053;
        }
        .stButton > button {
            background-color: #007BFF;
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 10px;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .result-box {
            background-color: #E8F0FE;
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            font-weight: bold;
            color: #1E40AF;
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("""
    <h1 style='text-align: center; color: #2E4053;'>KC-135 Work User Code Predictor</h1>
    <h4 style='text-align: left; color: #5D6D7E;'>Enter a discrepancy description to predict the work user code, 
                                                    system, and description as per TM 1C-135-06.</h4>
""", unsafe_allow_html=True)

# User input
discrepancy_text = st.text_area("Enter discrepancy description:", height=150)

# Predict button
if st.button("Predict Work User Code ✈️"):
    if discrepancy_text.strip():
        predicted_code = predict_discrepancy(discrepancy_text)
        st.markdown(f"<div class='result-box'>Predicted Work User Code: <br> <strong>{predicted_code}</strong></div>", unsafe_allow_html=True)
    else:
        st.error("Please enter a discrepancy description.")