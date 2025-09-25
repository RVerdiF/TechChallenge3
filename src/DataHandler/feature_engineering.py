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

def calculate_ema(data, window):
    """Calcula Média Móvel Exponencial."""
    return data.ewm(span=window, adjust=False).mean()

def calculate_macd(data, window_fast=12, window_slow=26, window_signal=9):
    """Calcula Moving Average Convergence Divergence (MACD)."""
    ema_fast = calculate_ema(data, window_fast)
    ema_slow = calculate_ema(data, window_slow)
    macd = ema_fast - ema_slow
    signal = calculate_ema(macd, window_signal)
    return macd, signal

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """Calcula Bandas de Bollinger."""
    sma = calculate_sma(data, window)
    std_dev = data.rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band

def calculate_stochastic_oscillator(data_close, data_high, data_low, window=14):
    """Calcula Oscilador Estocástico."""
    low_min = data_low.rolling(window=window).min()
    high_max = data_high.rolling(window=window).max()
    k_percent = 100 * ((data_close - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=3).mean()
    return k_percent, d_percent

def create_features(df, params={}):
    """
    Cria features para o modelo de ML.
    
    Args:
        df (pd.DataFrame): DataFrame com dados de preços
        params (dict): Dicionário com parâmetros para a engenharia de features
    
    Returns:
        pd.DataFrame: DataFrame com features adicionais
    """
    if df.empty:
        return df
    
    df_features = df.copy()
    
    # Parâmetros com valores padrão
    sma_window_1 = params.get('sma_window_1', 10)
    sma_window_2 = params.get('sma_window_2', 30)
    rsi_window = params.get('rsi_window', 14)
    ema_window_1 = params.get('ema_window_1', 12)
    ema_window_2 = params.get('ema_window_2', 26)
    macd_fast = params.get('macd_fast', 12)
    macd_slow = params.get('macd_slow', 26)
    macd_signal = params.get('macd_signal', 9)
    bollinger_window = params.get('bollinger_window', 20)
    stochastic_window = params.get('stochastic_window', 14)
    
    # Features de data
    df_features['month'] = df_features.index.month
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_month'] = df_features.index.day

    # Features de lag
    df_features['close_7_days_ago'] = df_features['Close'].shift(7)
    df_features['close_30_days_ago'] = df_features['Close'].shift(30)

    # Features de volume
    df_features['volume_change_pct'] = df_features['Volume'].pct_change() * 100

    # Médias Móveis Simples
    df_features[f'SMA_{sma_window_1}'] = calculate_sma(df_features['Close'], sma_window_1)
    df_features[f'SMA_{sma_window_2}'] = calculate_sma(df_features['Close'], sma_window_2)
    
    # RSI
    df_features['RSI'] = calculate_rsi(df_features['Close'], window=rsi_window)

    # EMA
    df_features[f'EMA_{ema_window_1}'] = calculate_ema(df_features['Close'], ema_window_1)
    df_features[f'EMA_{ema_window_2}'] = calculate_ema(df_features['Close'], ema_window_2)

    # MACD
    df_features['MACD'], df_features['MACD_signal'] = calculate_macd(df_features['Close'], window_fast=macd_fast, window_slow=macd_slow, window_signal=macd_signal)

    # Bollinger Bands
    df_features['Bollinger_Upper'], df_features['Bollinger_Lower'] = calculate_bollinger_bands(df_features['Close'], window=bollinger_window)

    # Stochastic Oscillator
    df_features['Stochastic_K'], df_features['Stochastic_D'] = calculate_stochastic_oscillator(df_features['Close'], df_features['High'], df_features['Low'], window=stochastic_window)
    
    # Target: 1 se preço de amanhã > preço de hoje, 0 caso contrário
    df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
    
    # Remove linhas com NaN
    df_features = df_features.dropna()
    
    return df_features