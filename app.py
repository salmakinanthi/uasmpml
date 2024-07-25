import streamlit as st
import joblib
import os
import requests
import pandas as pd

# URLs to the files on GitHub
MODEL_URL = "https://raw.githubusercontent.com/salmakinanthi/uasmpml/main/model.pkl"
ENCODERS_URL = "https://raw.githubusercontent.com/salmakinanthi/uasmpml/main/label_encoders.pkl"

# Local paths to downloaded files
MODEL_LOCAL_PATH = "model.pkl"
ENCODERS_LOCAL_PATH = "label_encoders.pkl"

def download_file(url, local_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        with open(local_path, 'wb') as f:
            f.write(response.content)
        st.write(f"Successfully downloaded {local_path}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading {local_path}: {e}")
    except IOError as e:
        st.error(f"Error saving {local_path}: {e}")

# Download the model and encoders if not available locally
if not os.path.isfile(MODEL_LOCAL_PATH):
    st.write("Downloading model...")
    download_file(MODEL_URL, MODEL_LOCAL_PATH)

if not os.path.isfile(ENCODERS_LOCAL_PATH):
    st.write("Downloading label encoders...")
    download_file(ENCODERS_URL, ENCODERS_LOCAL_PATH)

# Load the model and label encoders
label_encoders = {}
try:
    model = joblib.load(MODEL_LOCAL_PATH)
    label_encoders = joblib.load(ENCODERS_LOCAL_PATH)
    st.write("Model and label encoders loaded successfully.")
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
except joblib.externals.loky.process_executor.BrokenProcessPool as e:
    st.error(f"Error loading model: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

def preprocess_data(data):
    # Mengubah data ke DataFrame
    df = pd.DataFrame([data])

    # Encoding fitur kategorikal
    for feature, le in label_encoders.items():
        if feature in df.columns:
            df[feature] = le.transform(df[feature].astype(str))
    
    # Tambahkan kolom dummy jika model menggunakan kolom kategori yang sudah diencoding
    categorical_features = [
        'gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status'
    ]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Pastikan urutan dan nama kolom sesuai dengan yang digunakan model
    expected_columns = model.feature_names_in_  # Nama-nama fitur yang diharapkan model
    df = df.reindex(columns=expected_columns, fill_value=0)
    
    return df

def main():
    st.title('Stroke Prediction App')
    
    st.write("Masukkan data pasien untuk prediksi stroke:")
    
    # Form input untuk data pasien
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
        # Buat dictionary dari input
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
        
        # Preprocessing data
        try:
            df_preprocessed = preprocess_data(data)
            
            # Buat prediksi
            prediction = model.predict(df_preprocessed)[0]
            
            # Tampilkan hasil prediksi
            if prediction == 1:
                st.write("Predicted: Patient has a stroke.")
            else:
                st.write("Predicted: Patient does not have a stroke.")
        
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    main()
