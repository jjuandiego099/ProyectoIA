import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn

# Cargar modelos y scaler
scaler = joblib.load("scaler.joblib")
model = joblib.load("random_forest_balanced.joblib")

# Configurar estilo de fondo más claro
st.markdown(
    """
    <style>
        body {
            background-color: #f#D3D3D3;
        }
         div[data-baseweb="select"] div[role="option"]:hover {
        background-color: #FF0000 !important; /* Rojo para la opción al hacer hover */
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título y subtítulo
st.title("Predicción de cáncer de pulmón")
st.subheader("Juan Diego Chaparro y Juan José Vargas")

# Introducción
st.write("Esta aplicación permite predecir la probabilidad de padecer cáncer de pulmón en función de diferentes factores de riesgo. Ingrese sus datos a continuación para obtener una predicción.")

# Imagen
st.image("https://smart.servier.com/wp-content/uploads/2016/10/Embolie_pulmonaire_07.png")

# Entradas del usuario
age = st.slider("Edad", 30, 84, 50)
energy_level = st.text_input("Nivel de energía", "50.0")
oxygen_saturation = st.text_input("Saturación de oxígeno", "95.0")

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

data.update({feature_model: [1 if categorical_inputs[feature_display] == "Sí" else 0] 
              for feature_display, feature_model in zip(categorical_features_display, categorical_features_model) if feature_display != "Género"})

data["Gender"] = [1 if categorical_inputs["Género"] == "Hombre" else 0]

df = pd.DataFrame(data)

# Aplicar normalización solo a Age, Energy_Level y Oxygen_Saturation
numerical_features = ["Age", "Energy_Level", "Oxygen_Saturation"]
df[numerical_features] = scaler.transform(df[numerical_features])

# Asegurar que el DataFrame tenga todas las columnas esperadas en el mismo orden
expected_order = list(model.feature_names_in_)
df = df[expected_order]

# Predicción
prediction = model.predict(df)[0]

# Mostrar resultado con color de fondo
if prediction == 0:
    st.markdown('<div style="background-color: green; color: white; padding: 10px; text-align: center;">✅ No tiene cáncer de pulmón</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="background-color: red; color: white; padding: 10px; text-align: center;">⚠️ Tiene cáncer de pulmón</div>', unsafe_allow_html=True)

# Línea separadora
st.markdown("---")

# Copyright
st.write("© UNAB 2025")
print("Predicción del modelo:", prediction)
print("Valores escalados:", df[numerical_features].values)