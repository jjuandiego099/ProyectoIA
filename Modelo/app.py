import streamlit as st
import pandas as pd
import joblib
import numpy as np

import os
import joblib

# Verifica si el archivo existe
if os.path.exists("scaler.joblib"):
    print("El archivo 'scaler.joblib' existe.")
    scaler = joblib.load("scaler.joblib")
else:
    print("El archivo 'scaler.joblib' no se encuentra en la ruta especificada.")

model = joblib.load("random_forest_balanced.joblib")

# Título y subtítulo
st.title("Predicción de cáncer de pulmón")
st.subheader("Juan Diego Chaparro y Juan José Vargas")

# Introducción
st.write("Esta aplicación permite predecir la probabilidad de padecer cáncer de pulmón en función de diferentes factores de riesgo. Ingrese sus datos a continuación para obtener una predicción.")

# Imagen
st.image("https://smart.servier.com/wp-content/uploads/2016/10/Embolie_pulmonaire_07.png")

# Entradas del usuario: usando sliders en lugar de inputs de texto
age = st.slider("Edad", 30, 84, 50)

# Reemplazar text_input por slider para el Nivel de energía y la Saturación de oxígeno
energy_level = st.slider("Nivel de energía", 24, 83, 50)  # Slider entre 0 y 100
oxygen_saturation = st.slider("Saturación de oxígeno", 89, 99, 95)  # Slider entre 80% y 100%

# Variables categóricas
categorical_features_display = [
    "Género", "Fuma", "Decoloración de dedos", "Estrés mental", "Exposición a la contaminación",
    "Enfermedad a largo plazo", "Debilidad inmunológica", "Problemas respiratorios", "Consumo de alcohol",
    "Molestias en la garganta", "Opresión en el pecho", "Antecedentes familiares", "Antecedentes familiares de tabaquismo", "Estrés e inmunidad"
]

categorical_features_model = [
    "Gender", "Smoking", "Finger_Discoloration", "Mental_Stress", "Exposure_To_Pollution",
    "Long_Term_Illness", "Immune_Weakness", "Breathing_Issue", "Alcohol_Consumption",
    "Throat_Discomfort", "Chest_Tightness", "Family_History", "Smoking_Family_History", "Stress_Immune"
]

categorical_inputs = {}
for feature_display in categorical_features_display:
    if feature_display == "Género":
        categorical_inputs[feature_display] = st.selectbox(feature_display, ["Hombre", "Mujer"])
    else:
        categorical_inputs[feature_display] = st.selectbox(feature_display, ["No", "Sí"])

# Convertir inputs categóricos a 0 y 1
data = {
    "Age": [age],
    "Energy_Level": [float(energy_level)],
    "Oxygen_Saturation": [float(oxygen_saturation)]
}

data.update({
    feature_model: [1 if categorical_inputs[feature_display] == "Sí" else 0] 
    for feature_display, feature_model in zip(categorical_features_display, categorical_features_model) 
    if feature_display != "Género"
})

# Convertir "Género" a 0 o 1
data["Gender"] = [1 if categorical_inputs["Género"] == "Hombre" else 0]

# Crear DataFrame
df = pd.DataFrame(data)

# Normalizar las características numéricas
numerical_features = ["Age", "Energy_Level", "Oxygen_Saturation"]
df[numerical_features] = scaler.transform(df[numerical_features])

# Asegurarse de que las columnas estén en el orden correcto que espera el modelo
expected_order = list(model.feature_names_in_)
df = df[expected_order]

# Predicción del modelo
prediction = model.predict(df)[0]

# Usar st.empty() para crear un espacio de mensaje dinámico
message_placeholder = st.empty()  # Este será el espacio que se actualizará


if prediction == 0:
    message_placeholder.write("✅ **No tiene cáncer de pulmón**")
else:
    message_placeholder.write("⚠️ **Tiene cáncer de pulmón**")
st.write("© UNAB 2025")  # Copyright si predice cáncer de pulmón


st.write("DataFrame de entrada:", df)

st.write("Predicción del modelo:", prediction)
print("Valores escalados:", df[numerical_features].values)