import sqlite3
import joblib
import json
import io
from datetime import datetime
from pathlib import Path

# Define o caminho para o banco de dados no mesmo diretório
DB_PATH = Path(__file__).parent / "models.db"

def init_db():
    """Inicializa o banco de dados e cria as tabelas se não existirem."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_models (
                username TEXT PRIMARY KEY,
                model_blob BLOB NOT NULL,
                metrics_json TEXT NOT NULL,
                feature_params_json TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_configs (
                username TEXT PRIMARY KEY,
                model_params_json TEXT NOT NULL,
                feature_params_json TEXT NOT NULL
            )
        """)
        conn.commit()

def save_model(username: str, model_object, metrics_dict: dict, feature_params: dict):
    """Serializa e salva o modelo, métricas e parâmetros de features do usuário no banco de dados."""
    # Serializa o modelo para bytes
    model_buffer = io.BytesIO()
    joblib.dump(model_object, model_buffer)
    model_blob = model_buffer.getvalue()

    # Serializa métricas e parâmetros para string JSON
    metrics_json = json.dumps(metrics_dict)
    feature_params_json = json.dumps(feature_params)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_models (username, model_blob, metrics_json, feature_params_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (username, model_blob, metrics_json, feature_params_json, datetime.now()))
        conn.commit()

def load_model(username: str):
    """Carrega e desserializa o último modelo e métricas do usuário do banco de dados."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT model_blob, metrics_json, feature_params_json FROM user_models WHERE username = ?
        """, (username,))
        row = cursor.fetchone()

        if row:
            model_blob, metrics_json, feature_params_json = row

            # Desserializa o modelo
            model_buffer = io.BytesIO(model_blob)
            model = joblib.load(model_buffer)

            # Desserializa métricas e parâmetros
            metrics = json.loads(metrics_json)
            feature_params = json.loads(feature_params_json)
            
            # Adiciona os parâmetros de features ao dicionário de métricas para compatibilidade
            metrics['feature_params'] = feature_params

            return model, metrics
        else:
            return None, None

def save_config(username: str, config: dict):
    """Salva a configuração do usuário no banco de dados."""
    model_params_json = json.dumps(config.get("model_params", {}))
    feature_params_json = json.dumps(config.get("feature_params", {}))

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_configs (username, model_params_json, feature_params_json)
            VALUES (?, ?, ?)
        """, (username, model_params_json, feature_params_json))
        conn.commit()

def load_config(username: str):
    """Carrega a configuração do usuário do banco de dados."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT model_params_json, feature_params_json FROM user_configs WHERE username = ?
        """, (username,))
        row = cursor.fetchone()

        if row:
            model_params = json.loads(row[0])
            feature_params = json.loads(row[1])
            return {
                "model_params": model_params,
                "feature_params": feature_params
            }
        else:
            # Default values if no config is found
            return {
                "model_params": {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': -1,
                    'num_leaves': 31,
                    'reg_alpha': 0.0,
                    'reg_lambda': 0.0
                },
                "feature_params": {
                    'sma_window_1': 10,
                    'sma_window_2': 30,
                    'rsi_window': 14,
                    'ema_window_1': 12,
                    'ema_window_2': 26,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'bollinger_window': 20,
                    'stochastic_window': 14
                }
            }