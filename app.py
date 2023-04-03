import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

import pickle

# Load the trained model
with open('catboost_heart.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset
df = pd.read_csv('data\heart.csv')

# Add a title and a brief description of the app
st.title("Heart Disease Prediction App")
st.write("This app predicts the likelihood of a patient having heart disease based on various factors.")

# Display a sample of the dataset
st.write("Here's a sample of the dataset:")
st.write(df.head())    

# Define a function to preprocess the user input
def preprocess_input(age, sex, cp_type, bp, chol, fasting_bs, ecg, max_hr, ex_angina, oldpeak, st_slope):
    # Create a dictionary of the user input
    input_dict = {'Age': age, 'Sex': sex, 'ChestPainType': cp_type, 'RestingBP': bp, 'Cholesterol': chol,
                  'FastingBS': fasting_bs, 'RestingECG': ecg, 'MaxHR': max_hr, 'ExerciseAngina': ex_angina,
                  'Oldpeak': oldpeak, 'ST_Slope': st_slope}
    # Convert the dictionary into a pandas DataFrame
    input_df = pd.DataFrame.from_dict([input_dict])
    return input_df

# Define the Streamlit app
def app():
    # Define the input fields
    st.title('Heart Disease Prediction')
    age = st.slider('Age', min_value=0, max_value=120, step=1)
    sex = st.radio('Sex', ['Male', 'Female'])
    cp_type = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    bp = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, step=1)
    chol = st.number_input('Cholesterol', min_value=0, max_value=1000, step=1)
    fasting_bs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
    ecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
    max_hr = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, step=1)
    ex_angina = st.radio('Exercise-Induced Angina', ['Yes', 'No'])
    oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, step=0.1)
    st_slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    
    # Preprocess the user input
    input_df = preprocess_input(age, sex, cp_type, bp, chol, fasting_bs, ecg, max_hr, ex_angina, oldpeak, st_slope)
    
    # Define the predict button
    if st.button('Predict'):
        # Use the model to make predictions on the input data
        pred = model.predict(input_df)
        # Display the predicted result
        if pred[0] == 0:
            st.write('No heart disease')
        else:
            st.write('Heart disease present')

# Run the app
if __name__ == '__main__':
    app()
