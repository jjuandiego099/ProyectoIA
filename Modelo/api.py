# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Cargar modelo y scaler
ruta = os.path.join(os.path.dirname(__file__), "scaler.joblib")
ruta2 = os.path.join(os.path.dirname(__file__), "random_forest_balanced.joblib")
scaler = joblib.load(ruta)
model = joblib.load(ruta2)

# Lista esperada de columnas (orden correcto)
expected_order = list(model.feature_names_in_)

# Crear la app
app = FastAPI(title="API Cáncer de Pulmón")

# Modelo de datos para entrada
class PatientData(BaseModel):
    Age: int
    Energy_Level: float
    Oxygen_Saturation: float
    Gender: int  # 1 = Hombre, 0 = Mujer
    Smoking: int
    Finger_Discoloration: int
    Mental_Stress: int
    Exposure_To_Pollution: int
    Long_Term_Illness: int
    Immune_Weakness: int
    Breathing_Issue: int
    Alcohol_Consumption: int
    Throat_Discomfort: int
    Chest_Tightness: int
    Family_History: int
    Smoking_Family_History: int
    Stress_Immune: int
@app.get("/")
def root():
    return {"message": "API de predicción de cáncer de pulmón funcionando correctamente"}

@app.post("/app")
def predict(data: PatientData):
    # Convertir entrada a DataFrame
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Normalizar variables numéricas
    numerical_features = ["Age", "Energy_Level", "Oxygen_Saturation"]
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Reordenar columnas al orden esperado por el modelo
    df = df[expected_order]

    # Hacer predicción
    prediction = int(model.predict(df)[0])
    result = "Tiene cáncer de pulmón" if prediction == 1 else "No tiene cáncer de pulmón"

    return {
        "prediction": prediction,
        "result": result
    }
