import streamlit as st
import joblib
import pandas as pd
import os

# Tentukan jalur absolut untuk file model dan encoder
model_path = 'D:/kuliah/sem 4/MPML/STROKe/model.pkl'
encoder_path = 'D:/kuliah/sem 4/MPML/STROKe/label_encoders.pkl'

# Memastikan model dan encoder ada
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")
if not os.path.isfile(encoder_path):
    raise FileNotFoundError(f"Encoder file {encoder_path} does not exist.")

# Memuat model dan encoder
model = joblib.load(model_path)
label_encoders = joblib.load(encoder_path)

def preprocess_data(data):
    # Mengubah data ke DataFrame
    df = pd.DataFrame([data])

    # Encoding fitur kategorikal
    for feature, le in label_encoders.items():
        if feature in df.columns:
            df[feature] = le.transform(df[feature].astype(str))
    
    # Tambahkan kolom dummy jika model menggunakan kolom kategori yang sudah diencoding
    df = pd.get_dummies(df, columns=[
        'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
    ], drop_first=True)
    
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
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        
        # Preprocessing data
        df_preprocessed = preprocess_data(data)
        
        # Buat prediksi
        prediction = model.predict(df_preprocessed)[0]
        
        # Tampilkan hasil prediksi
        if prediction == 1:
            st.write("Predicted: Patient has a stroke.")
        else:
            st.write("Predicted: Patient does not have a stroke.")

if __name__ == "__main__":
    main()
