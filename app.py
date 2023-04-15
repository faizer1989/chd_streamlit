import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from PIL import Image

import pickle

# Load the trained model
with open('catboost_heart.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset
df = pd.read_csv('heart.csv')

# Load an image file from your local machine or a URL
school_image = Image.open('school_logo.jpg')  # Replace with the correct path to your image file or a URL
app_image = Image.open('app_logo.jpg')  # Replace with the correct path to your image file or a URL
# Add a title and a brief description of the app


# Create responsive layout columns
left_column, right_column = st.columns(2)

# Insert the image in the middle column
with right_column:
    st.image(app_image, width=150)

with left_column:
    st.title("OBISCOPE")

st.write("___")
left_column1, right_column1 = st.columns(2)

# Insert the image in the middle column
with right_column1:
    st.image(school_image, width=70)

with left_column1:
    st.markdown("<i>The heart disease prediction app has been developed by 'Precious Heart' team from Our Lady's Sec. Sch. Nnobi, Nigeria for Technovation World Challenge 2023.</i>", unsafe_allow_html=True)
    
st.write("___")   
    
  


# Display a sample of the dataset
st.write("Dataset example that was used to build AI prediction model:")
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
    st.title("Enter the vital signs to predict CAD")
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
    
    if st.button('Predict'):
        pred = model.predict(input_df)
        if pred[0] == 0:
            st.write('No heart disease')
            st.write('Great! You have a low risk of heart disease! Keep maintaining a healthy lifestyle by eating a balanced diet, exercising regularly, managing stress, and avoiding smoking and excessive alcohol consumption.')

        elif pred[0] == 1:
            st.write('Heart disease detected')
            st.write('It appears that you have a risk of heart disease. It is crucial to consult with a medical professional for a proper diagnosis and management plan. In the meantime, consider making lifestyle changes such as adopting a heart-healthy diet, increasing physical activity, managing stress, quitting smoking, and limiting alcohol intake.')

if __name__ == '__main__':
    app()
