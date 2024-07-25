import streamlit as st
import joblib
import requests
import pandas as pd
from io import BytesIO

# URLs to the files on GitHub
MODEL_URL = "https://raw.githubusercontent.com/salmakinanthi/uasmpml/master/best_model.pkl"

def load_file_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading file from {url}: {e}")
        return None

# Load the model directly from URL
model_file = load_file_from_url(MODEL_URL)

if model_file is not None:
    try:
        model, feature_names = joblib.load(model_file)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
else:
    model = None

def preprocess_data(data):
    if model is None:
        st.error("Model not loaded.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Convert data to DataFrame
    df = pd.DataFrame([data])
    
    # Add dummy columns if the model uses encoded categorical columns
    categorical_features = [
        'gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status'
    ]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Ensure column order and names match what the model expects
    df = df.reindex(columns=feature_names, fill_value=0)
    
    return df

def main():
    st.title('Stroke Prediction App')
    
    st.write("Masukkan data pasien untuk prediksi stroke:")
    
    # Form input for patient data
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.number_input('Age', min_value=0)
    hypertension = st.selectbox('Hypertension', [0, 1])
    heart_disease = st.selectbox('Heart Disease', [0, 1])
    ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
    work_type = st.selectbox('Work Type', ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
    residence_type = st.selectbox('Residence Type', ['Rural', 'Urban'])
    avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0)
    bmi = st.number_input('BMI', min_value=0.0)
    smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
    
    submit_button = st.button('Predict')

    if submit_button:
        if model is None:
            st.error("Model is not properly loaded.")
            return
        
        # Create dictionary from input
        data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        
        # Preprocess data
        try:
            df_preprocessed = preprocess_data(data)
            
            # Make prediction
            prediction = model.predict(df_preprocessed)[0]
            
            # Display result
            if prediction == 1:
                st.write("Predicted: Patient has a stroke.")
            else:
                st.write("Predicted: Patient does not have a stroke.")
        
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    main()
