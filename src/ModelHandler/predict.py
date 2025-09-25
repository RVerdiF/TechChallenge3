import pandas as pd
import joblib
import os
import json
from pathlib import Path

MODEL_PATH = Path('src/ModelHandler/lgbm_model.pkl')
METRICS_PATH = Path('src/ModelHandler/metrics.json')

def make_prediction(input_data):
    """
    Faz previsão usando o modelo salvo.
    
    Args:
        input_data (pd.DataFrame): Dados com features calculadas
    
    Returns:
        tuple: (Previsão, Confiança)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Modelo não encontrado. Execute o treinamento primeiro.")
    
    # Carrega modelo
    model = joblib.load(MODEL_PATH)
    
    # Carrega lista de features do treinamento
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    feature_cols = metrics['features']
    
    # Verifica se todas as features estão presentes
    missing_features = [col for col in feature_cols if col not in input_data.columns]
    if missing_features:
        raise ValueError(f"Features faltando: {missing_features}")
    
    # Seleciona apenas as features necessárias
    X = input_data[feature_cols].iloc[-1:]  # Última linha
    
    # Faz previsão de probabilidade
    prediction_proba = model.predict_proba(X)[0]
    prediction = prediction_proba.argmax()
    confidence = prediction_proba[prediction]
    
    return int(prediction), float(confidence)