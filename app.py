import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Load the trained model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# UI Styling for dark mode
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #27ae60;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1e1e1e;
        border: 2px solid #3e3e3e;
        margin-top: 20px;
        text-align: center;
        font-size: 20px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Customer Churn Prediction using ANN</div>', unsafe_allow_html=True)

# Input form
with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        age = st.slider('Age', 18, 92)
        balance = st.number_input('Balance')
        tenure = st.slider('Tenure (Years)', 0, 10)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1])

    with col2:
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        credit_score = st.number_input('Credit Score')
        estimated_salary = st.number_input('Estimated Salary')
        num_of_products = st.slider('Number of Products', 1, 4)
        is_active_member = st.selectbox('Active Member', [0, 1])

    # Submit button
    submit_button = st.form_submit_button(label='Predict')

# Prediction logic
if submit_button:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encoding for Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Display result
    st.markdown(f"""
    <div class="prediction-box">
        <strong>Churn Probability:</strong> {prediction_proba:.2f} <br>
        {'The customer is likely to churn.' if prediction_proba > 0.5 else 'The customer is not likely to churn.'}
    </div>
    """, unsafe_allow_html=True)
