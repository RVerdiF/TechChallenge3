import pandas as pd
import joblib
import os

MODEL_PATH = 'lgbm_model.pkl'

def make_prediction(input_data):
    """
    Faz previsão usando o modelo salvo.
    
    Args:
        input_data (pd.DataFrame): Dados com features calculadas
    
    Returns:
        int: Previsão (0 para Queda, 1 para Alta)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Modelo não encontrado. Execute o treinamento primeiro.")
    
    # Carrega modelo
    model = joblib.load(MODEL_PATH)
    
    # Features esperadas pelo modelo
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_30', 'RSI']
    
    # Verifica se todas as features estão presentes
    missing_features = [col for col in feature_cols if col not in input_data.columns]
    if missing_features:
        raise ValueError(f"Features faltando: {missing_features}")
    
    # Seleciona apenas as features necessárias
    X = input_data[feature_cols].iloc[-1:]  # Última linha
    
    # Faz previsão
    prediction = model.predict(X)[0]
    
    return int(prediction)