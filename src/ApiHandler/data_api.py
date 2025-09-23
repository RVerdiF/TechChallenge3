import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_btc_data(ticker="BTC-USD", start_date=None, end_date=None):
    """
    Obtém dados históricos do Bitcoin usando yfinance.
    
    Args:
        ticker (str): Símbolo do ticker (padrão: "BTC-USD")
        start_date (str): Data de início no formato "YYYY-MM-DD"
        end_date (str): Data de fim no formato "YYYY-MM-DD"
    
    Returns:
        pd.DataFrame: DataFrame com colunas Open, High, Low, Close, Volume
    """
    try:
        # Define datas padrão se não fornecidas
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Baixa os dados
        btc = yf.Ticker(ticker)
        data = btc.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"Nenhum dado encontrado para {ticker} no período especificado")
        
        # Seleciona apenas as colunas necessárias
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.index.name = 'Date'
        
        return data
        
    except Exception as e:
        print(f"Erro ao obter dados: {e}")
        return pd.DataFrame()