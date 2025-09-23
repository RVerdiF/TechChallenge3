import pandas as pd
import sqlite3
import os

DB_PATH = "btc_prices.db"

def init_database():
    """Inicializa o banco de dados SQLite com a tabela prices."""
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
    Salva DataFrame na tabela prices, evitando duplicação.
    
    Args:
        df (pd.DataFrame): DataFrame com dados de preços
    """
    if df.empty:
        return
    
    init_database()
    conn = sqlite3.connect(DB_PATH)
    
    # Prepara os dados
    df_copy = df.copy()
    df_copy.index = df_copy.index.strftime('%Y-%m-%d')
    df_copy.columns = [col.lower() for col in df_copy.columns]
    
    # Salva usando replace para evitar duplicatas
    df_copy.to_sql('prices', conn, if_exists='replace', index_label='date')
    
    conn.close()

def load_data():
    """
    Carrega todos os dados da tabela prices.
    
    Returns:
        pd.DataFrame: DataFrame com dados históricos
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        df = pd.read_sql_query("SELECT * FROM prices ORDER BY date", conn)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.columns = [col.capitalize() for col in df.columns]
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()