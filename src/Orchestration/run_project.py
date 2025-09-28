"""
Script principal para executar o projeto de previsão de BTC.
"""

import argparse
import src.ApiHandler.data_api as data_api
import src.DataHandler.data_handler as data_handler
import src.ModelHandler.train_model as train_model
from datetime import datetime, timedelta

from src.LogHandler.log_config import get_logger
from src.config import USER_DATA_DIR

logger = get_logger(__name__)

def setup_project(username="Admin", update_only=True):
    """Configura o projeto: coleta dados e treina modelo."""
    logger.info(f"=== Iniciando pipeline de dados e treino para o usuário {username} ===")

    MODEL_PATH = USER_DATA_DIR / username / "lgbm_model.pkl"
    METRICS_PATH = USER_DATA_DIR / username / "metrics.json"
    
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
    logger.info("Execute 'streamlit run main.py' para abrir o dashboard.")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup project for a given user.")
    parser.add_argument("--user", type=str, default="Admin", help="Username to setup the project for.")
    parser.add_argument("--full-reload", action="store_true", help="Force a full reload of the data.")
    args = parser.parse_args()

    setup_project(username=args.user, update_only=not args.full_reload)
