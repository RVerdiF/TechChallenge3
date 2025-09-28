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

def setup_project(update_only=True):
    """Configura o projeto: coleta dados e treina modelo."""
    logger.info("=== Iniciando pipeline de dados e treino ===")
    
    # 1. Determina o range de datas para a coleta
    logger.info("1. Verificando dados existentes para atualização...")
    existing_data = data_handler.load_data()
    
    end_date = datetime.now().strftime("%Y-%m-%d")

    if update_only and not existing_data.empty:
        last_date = existing_data.index.max()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"Última data no banco: {last_date.date()}. Buscando dados a partir de {start_date}.")
    else:
        start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
        logger.info(f"Nenhum dado local ou 'update_only=False'. Buscando dados dos últimos 3 anos.")

    # 2. Coleta dados novos
    logger.info(f"2. Coletando dados de {start_date} a {end_date}...")
    new_df = data_api.get_btc_data(start_date=start_date, end_date=end_date)
    
    if new_df.empty:
        logger.warning("Nenhum dado novo foi coletado. O modelo não será retreinado.")
        return False
    
    logger.info(f"{len(new_df)} novos registros coletados.")
    
    # 3. Salva dados
    logger.info("3. Salvando novos dados no banco...")
    data_handler.save_data(new_df)
    
    # 4. Treina modelo com dados atualizados
    logger.info("4. Treinando modelo com o dataset completo...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_model.train_and_save_model(model_path=MODEL_PATH, metrics_path=METRICS_PATH)
    
    logger.info("=== Pipeline finalizado com sucesso! ===")
    logger.info("Execute 'streamlit run dashboard.py' para abrir o dashboard.")
    
    return True

if __name__ == "__main__":
    # Por padrão, o script agora apenas atualiza os dados e retreina.
    # Para forçar uma recarga completa, chame setup_project(update_only=False)
    setup_project()
