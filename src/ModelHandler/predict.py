import pandas as pd
from src.LogHandler.log_config import get_logger

logger = get_logger(__name__)

def make_prediction(input_data: pd.DataFrame, model, metrics: dict):
    """
    Faz previsão usando o modelo carregado.
    
    Args:
        input_data (pd.DataFrame): Dados com as features calculadas.
        model: Objeto do modelo treinado e carregado.
        metrics (dict): Dicionário de métricas que contém a lista de features usadas no treinamento.
    
    Returns:
        tuple: (Previsão, Confiança)
    """
    if model is None or metrics is None:
        logger.error("Model or metrics not provided to make_prediction function.")
        raise ValueError("Modelo ou métricas não foram fornecidos. Treine um modelo primeiro.")

    feature_cols = metrics.get('features')
    if not feature_cols:
        logger.error("'features' key not found in metrics dictionary.")
        raise ValueError("Lista de features não encontrada nas métricas. Treine o modelo novamente.")

    # Garante que a última linha dos dados de entrada seja usada
    latest_data = input_data.iloc[-1:]

    # Verifica se todas as features necessárias estão presentes nos dados de entrada
    missing_features = [col for col in feature_cols if col not in latest_data.columns]
    if missing_features:
        raise ValueError(f"Features faltando nos dados de entrada: {missing_features}")
    
    # Seleciona apenas as features usadas no treinamento, na ordem correta
    X = latest_data[feature_cols]
    
    # Faz a previsão de probabilidade
    logger.info(f"Making prediction with {len(feature_cols)} features.")
    prediction_proba = model.predict_proba(X)[0]
    prediction = prediction_proba.argmax()
    confidence = prediction_proba[prediction]
    
    logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
    return int(prediction), float(confidence)
