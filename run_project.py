#!/usr/bin/env python3
"""
Script principal para executar o projeto de previsão de BTC.
"""

import src.ApiHandler.data_api as data_api
import src.DataHandler.data_handler as data_handler
import src.ModelHandler.train_model as train_model
from datetime import datetime, timedelta

def setup_project():
    """Configura o projeto: coleta dados e treina modelo."""
    print("=== Configuração do Projeto BTC Prediction ===")
    
    # 1. Coleta dados
    print("\n1. Coletando dados históricos...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
    
    df = data_api.get_btc_data(start_date=start_date, end_date=end_date)
    
    if df.empty:
        print("Erro: Não foi possível obter dados.")
        return False
    
    print(f"Dados coletados: {len(df)} registros")
    
    # 2. Salva dados
    print("\n2. Salvando dados no banco...")
    data_handler.save_data(df)
    print("Dados salvos com sucesso!")
    
    # 3. Treina modelo
    print("\n3. Treinando modelo...")
    train_model.train_and_save_model()
    
    print("\n=== Projeto configurado com sucesso! ===")
    print("Execute 'streamlit run dashboard.py' para abrir o dashboard.")
    
    return True

if __name__ == "__main__":
    setup_project()