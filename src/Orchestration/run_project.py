"""
Script principal para executar o projeto de previsão de BTC.
"""

import src.ApiHandler.data_api as data_api
import src.DataHandler.data_handler as data_handler
import src.ModelHandler.train_model as train_model
from datetime import datetime, timedelta
from pathlib import Path
from src.LogHandler.log_config import get_logger

logger = get_logger(__name__)

# Define paths for the default user 'Admin'
MODEL_PATH = Path("user_data/Admin/model.pkl")
METRICS_PATH = Path("user_data/Admin/metrics.json")

def setup_project():
    """Configura o projeto: coleta dados e treina modelo."""
    logger.info("=== Configuração do Projeto BTC Prediction ===")
    
    # 1. Coleta dados
    logger.info("1. Coletando dados históricos...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
    
    df = data_api.get_btc_data(start_date=start_date, end_date=end_date)
    
    if df.empty:
        logger.error("Erro: Não foi possível obter dados.")
        return False
    
    logger.info(f"Dados coletados: {len(df)} registros")
    
    # 2. Salva dados
    logger.info("2. Salvando dados no banco...")
    data_handler.save_data(df)
    logger.info("Dados salvos com sucesso!")
    
    # 3. Treina modelo
    logger.info("3. Treinando modelo...")
    # Ensure the parent directory for model/metrics exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_model.train_and_save_model(model_path=MODEL_PATH, metrics_path=METRICS_PATH)
    
    logger.info("=== Projeto configurado com sucesso! ===")
    logger.info("Execute 'streamlit run dashboard.py' para abrir o dashboard.")
    
    return True

if __name__ == "__main__":
    setup_project()