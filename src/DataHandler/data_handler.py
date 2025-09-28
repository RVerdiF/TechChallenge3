import pandas as pd
import sqlite3
from src.LogHandler.log_config import get_logger

logger = get_logger(__name__)

from src.config import BTC_PRICES_DB

DB_PATH = BTC_PRICES_DB

def init_database():
    """Inicializa o banco de dados SQLite com a tabela prices."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_data(df):
    """
    Salva DataFrame na tabela prices, apenas com dados novos (append).
    
    Args:
        df (pd.DataFrame): DataFrame com dados de preços
    """
    if df.empty:
        return
    
    init_database()
    conn = sqlite3.connect(DB_PATH)
    
    df_copy = df.copy()
    df_copy.index = df_copy.index.strftime('%Y-%m-%d')
    df_copy.columns = [col.lower() for col in df_copy.columns]

    existing_dates = pd.read_sql("SELECT date FROM prices", conn)['date'].tolist()
    
    df_to_append = df_copy[~df_copy.index.isin(existing_dates)]
    
    if not df_to_append.empty:
        df_to_append.to_sql('prices', conn, if_exists='append', index_label='date')
        logger.info(f"{len(df_to_append)} novos registros salvos.")
    else:
        logger.info("Nenhum registro novo para salvar.")
    
    conn.close()

def update_data(df, start_date, end_date):
    """
    Atualiza dados no banco para o período especificado, substituindo dados existentes.
    
    Args:
        df (pd.DataFrame): DataFrame com dados de preços
        start_date (str): Data inicial no formato 'YYYY-MM-DD'
        end_date (str): Data final no formato 'YYYY-MM-DD'
    """
    if df.empty:
        return
    
    init_database()
    conn = sqlite3.connect(DB_PATH)
    
    # Remove dados existentes no período
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prices WHERE date >= ? AND date <= ?", (start_date, end_date))
    
    # Insere novos dados
    df_copy = df.copy()
    df_copy.index = df_copy.index.strftime('%Y-%m-%d')
    df_copy.columns = [col.lower() for col in df_copy.columns]
    df_copy.to_sql('prices', conn, if_exists='append', index_label='date')
    
    conn.commit()
    conn.close()
    logger.info(f"Dados atualizados para o período {start_date} a {end_date}: {len(df)} registros.")

def drop_table():
    """
    Remove todos os dados da tabela prices.
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prices")
    conn.commit()
    conn.close()
    logger.info("Tabela prices limpa completamente.")

def load_data():
    """
    Carrega todos os dados da tabela prices.
    
    Returns:
        pd.DataFrame: DataFrame com dados históricos
    """
    if not DB_PATH.exists():
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        df = pd.read_sql_query("SELECT * FROM prices ORDER BY date", conn)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.columns = [col.capitalize() for col in df.columns]
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()
    finally:
        conn.close()