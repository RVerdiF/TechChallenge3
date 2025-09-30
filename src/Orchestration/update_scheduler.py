import time
from datetime import datetime, timedelta, date
from src.LogHandler.log_config import get_logger
import src.DataHandler.data_handler as data_handler
import src.ApiHandler.data_api as data_api

logger = get_logger(__name__)

LAST_UPDATE_KEY = "last_successful_data_update"

def get_last_update_date() -> date | None:
    """Lê a data da última atualização do banco de dados."""
    date_str = data_handler.get_metadata(LAST_UPDATE_KEY)
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

def set_last_update_date(update_date: date):
    """Registra a data da última atualização no banco de dados."""
    data_handler.set_metadata(LAST_UPDATE_KEY, update_date.strftime("%Y-%m-%d"))

def run_daily_data_update():
    """Executa a lógica de atualização de dados."""
    logger.info("Verificando necessidade de atualização diária de dados...")
    
    last_update = get_last_update_date()
    today = datetime.now().date()

    if last_update and last_update >= today:
        logger.info(f"Dados já estão atualizados para o dia de hoje ({today}). Próxima verificação em 1 hora.")
        return

    try:
        df_existing = data_handler.load_data()
        
        if not df_existing.empty:
            start_date = df_existing.index.max().date() + timedelta(days=1)
        else:
            start_date = date(2009, 1, 3)
        
        end_date = today

        if start_date > end_date:
            logger.info("Base de dados já contém os dados mais recentes. Nenhuma atualização necessária.")
            set_last_update_date(today)
            return

        logger.info(f"Buscando novos dados de {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
        
        df_new = data_api.get_btc_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        if not df_new.empty:
            data_handler.save_data(df_new)
            logger.info("Atualização de dados concluída com sucesso.")
        else:
            logger.info("Nenhum dado novo foi retornado pela API.")
        
        set_last_update_date(today)

    except Exception as e:
        logger.error(f"Falha na rotina de atualização diária de dados: {e}", exc_info=True)

def daily_update_task():
    """Tarefa que roda em segundo plano para verificar e atualizar os dados diariamente."""
    # Espera um pouco no início para a aplicação principal carregar
    time.sleep(10)
    while True:
        run_daily_data_update()
        # A tarefa dorme por 1 hora antes de verificar novamente
        time.sleep(3600)