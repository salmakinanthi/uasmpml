import streamlit as st
import joblib
import requests
import pandas as pd
from io import BytesIO

# URLs to the files on GitHub
MODEL_URL = "https://github.com/salmakinanthi/uasmpml/blob/master/app.py"
ENCODERS_URL = "https://github.com/salmakinanthi/uasmpml/blob/master/label_encoders.pkl"

def load_file_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading file from {url}: {e}")
        return None

# Load the model and label encoders directly from URL
model_file = load_file_from_url(MODEL_URL)
encoders_file = load_file_from_url(ENCODERS_URL)

if model_file is not None and encoders_file is not None:
    try:
        model = joblib.load(model_file)
        label_encoders = joblib.load(encoders_file)
        st.write("Model and label encoders loaded successfully.")
    except Exception as e:
        st.error(f"An error occurred while loading the model or encoders: {e}")
else:
    model = None
    label_encoders = {}

def preprocess_data(data):
    if model is None or not label_encoders:
        st.error("Model or label encoders not loaded.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Convert data to DataFrame
    df = pd.DataFrame([data])

    # Encoding categorical features
    for feature, le in label_encoders.items():
        if feature in df.columns:
            df[feature] = le.transform(df[feature].astype(str))
    
    # Add dummy columns if the model uses encoded categorical columns
    categorical_features = [
        'gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status'
    ]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Ensure column order and names match what the model expects
    expected_columns = model.feature_names_in_  # Feature names expected by the model
    df = df.reindex(columns=expected_columns, fill_value=0)
    
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
        if model is None or not label_encoders:
            st.error("Model or label encoders are not properly loaded.")
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
