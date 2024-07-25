import streamlit as st
import joblib
import pandas as pd
import os
import requests

# URLs to the files on GitHub
MODEL_URL = "https://raw.githubusercontent.com/salmakinanthi/uasmpml/main/model.pkl"
ENCODERS_URL = "https://raw.githubusercontent.com/salmakinanthi/uasmpml/main/label_encoders.pkl"
MODEL_LOCAL_PATH = ""D:\kuliah\sem 4\MPML\STROKe\model.pkl""
ENCODERS_LOCAL_PATH = ""D:\kuliah\sem 4\MPML\STROKe\label_encoders.pkl""

def download_file(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)

# Download the model and encoders if not available locally
if not os.path.isfile(MODEL_LOCAL_PATH):
    st.write("Downloading model...")
    download_file(MODEL_URL, MODEL_LOCAL_PATH)
    st.write("Model download complete.")

if not os.path.isfile(ENCODERS_LOCAL_PATH):
    st.write("Downloading label encoders...")
    download_file(ENCODERS_URL, ENCODERS_LOCAL_PATH)
    st.write("Label encoders download complete.")

# Load the model and label encoders
model = joblib.load(MODEL_LOCAL_PATH)
label_encoders = joblib.load(ENCODERS_LOCAL_PATH)

def preprocess_data(data):
    df = pd.DataFrame([data])
    for feature, le in label_encoders.items():
        if feature in df.columns:
            df[feature] = le.transform(df[feature].astype(str))
    df = pd.get_dummies(df, columns=[
        'gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status'
    ], drop_first=True)
    expected_columns = model.feature_names_in_
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

def main():
    st.title('Stroke Prediction App')
    st.write("Masukkan data pasien untuk prediksi stroke:")
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
        df_preprocessed = preprocess_data(data)
        prediction = model.predict(df_preprocessed)[0]
        if prediction == 1:
            st.write("Predicted: Patient has a stroke.")
        else:
            st.write("Predicted: Patient does not have a stroke.")

if __name__ == "__main__":
    main()
