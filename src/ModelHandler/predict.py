import pandas as pd
import joblib
import os
import json
from pathlib import Path

def make_prediction(input_data, model_path):
    """
    Faz previsão usando o modelo salvo.
    
    Args:
        input_data (pd.DataFrame): Dados com features calculadas
    
    Returns:
        tuple: (Previsão, Confiança)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("Modelo não encontrado. Execute o treinamento primeiro.")
    
    # Carrega modelo
    model = joblib.load(model_path)
    
    # Carrega lista de features do treinamento
    metrics_path = model_path.parent / 'metrics.json'
    with open(metrics_path, 'r') as f:
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