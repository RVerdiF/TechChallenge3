import pandas as pd
import numpy as np

def calculate_sma(data, window):
    """Calcula Média Móvel Simples."""
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calcula Índice de Força Relativa (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features(df):
    """
    Cria features para o modelo de ML.
    
    Args:
        df (pd.DataFrame): DataFrame com dados de preços
    
    Returns:
        pd.DataFrame: DataFrame com features adicionais
    """
    if df.empty:
        return df
    
    df_features = df.copy()
    
    # Médias Móveis Simples
    df_features['SMA_10'] = calculate_sma(df_features['Close'], 10)
    df_features['SMA_30'] = calculate_sma(df_features['Close'], 30)
    
    # RSI
    df_features['RSI'] = calculate_rsi(df_features['Close'])
    
    # Target: 1 se preço de amanhã > preço de hoje, 0 caso contrário
    df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
    
    # Remove linhas com NaN
    df_features = df_features.dropna()
    
    return df_features