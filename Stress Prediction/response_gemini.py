import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables and configure Google Gemini API key
load_dotenv()
genai.configure(api_key='AIzaSyDOep_JUhakDhjhfw_-hT_MMCZ5fF1uRwo')

# Load  trained model
model = load_model('C:/Users/Windows/Downloads/Project/stressprediction.keras')

# Function for predictions
def predict_condition(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

# Function to generate a suggestion using Google Gemini based on stress level
def get_gemini_suggestion(stress_level):
    if stress_level == "Low Level":
        query = "Provide general well-being tips for someone with no signs of stress."
    elif stress_level == "Mid Level":
        query = "Give relaxation and mental wellness advice for someone showing middle level of stress."
    elif stress_level == "High Level":
        query = "Provide tips for managing stress under time pressure, such as relaxation techniques and time management."
    
    prompt = f"""
    You are an expert in stress management. Based on a prediction of '{stress_level}', provide personalized suggestions for handling stress.
    Query: {query}
    """
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt])
    
    return response.text

# Streamlit app setup
st.title('Stress Prediction and Management Suggestions')

st.write("""
This app predicts your stress condition based on heart rate variability (HRV) data and provides personalized stress management advice.
Enter the required parameters to get a prediction and corresponding advice.
""")

# Input fields for user to input the features
MEAN_RR = st.number_input("Mean of RR intervals (in ms)", min_value=0.0, value=0.0)
RMSSD = st.number_input("Root mean square of successive RR interval differences (in ms)", min_value=0.0, value=0.0)
pNN25 = st.number_input("Percentage of successive RR intervals that differ by more than 25 ms", min_value=0.0, value=0.0)
pNN50 = st.number_input("Percentage of successive RR intervals that differ by more than 50 ms", min_value=0.0, value=0.0)
LF = st.number_input("Low frequency power (LF) (0.04 - 0.15 Hz)", min_value=0.0, value=0.0)
HF = st.number_input("High frequency power (HF) (0.15 - 0.4 Hz)", min_value=0.0, value=0.0)
LF_HF = st.number_input("Ratio of LF to HF", min_value=0.0, value=0.0)

# Collect the inputs into a list
input_features = [MEAN_RR, RMSSD, pNN25, pNN50, LF, HF, LF_HF]

# Predict button
if st.button('Predict'):
    # Scale input features before passing them to the model
    scaler = StandardScaler()
    input_features_scaled = scaler.fit_transform([input_features])

    # Call the prediction function
    condition = predict_condition(input_features_scaled)

    # Map condition to readable label
    condition_mapping = {0: "Low Level", 1: "Mid Level", 2: "High Level"}
    predicted_condition = condition_mapping[condition]
    st.write(f"The predicted stress condition is: {predicted_condition}")

    # Generate suggestion using Gemini API based on the predicted condition
    gemini_suggestion = get_gemini_suggestion(predicted_condition)
    st.subheader("Personalized Stress Management Suggestion")
    st.write(gemini_suggestion)
