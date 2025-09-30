import sqlite3
import joblib
import json
import io
from datetime import datetime
from pathlib import Path

# Define o caminho para o banco de dados no mesmo diretório
DB_PATH = Path(__file__).parent / "models.db"

from stable_baselines3 import PPO

def init_db():
    """Inicializa o banco de dados e cria/atualiza as tabelas se necessário."""
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
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rl_models (
                model_name TEXT PRIMARY KEY,
                model_blob BLOB NOT NULL,
                feature_params_json TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)

        # --- Lógica de Migração ---
        # Migração para a tabela rl_models
        cursor.execute("PRAGMA table_info(rl_models)")
        rl_columns = [info[1] for info in cursor.fetchall()]
        if 'metrics_json' not in rl_columns:
            cursor.execute("ALTER TABLE rl_models ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'")
        if 'rl_params_json' not in rl_columns:
            cursor.execute("ALTER TABLE rl_models ADD COLUMN rl_params_json TEXT NOT NULL DEFAULT '{}'")
        
        # Migração para a tabela user_configs
        cursor.execute("PRAGMA table_info(user_configs)")
        config_columns = [info[1] for info in cursor.fetchall()]
        if 'rl_params_json' not in config_columns:
            cursor.execute("ALTER TABLE user_configs ADD COLUMN rl_params_json TEXT NOT NULL DEFAULT '{}'")

        conn.commit()

def save_rl_model(model, feature_params: dict, metrics_dict: dict, rl_params: dict, model_name: str = "default_ppo"):
    """Serializa e salva um modelo de RL, seus parâmetros e métricas no banco de dados."""
    buffer = io.BytesIO()
    model.save(buffer)
    buffer.seek(0)
    model_blob = buffer.read()

    feature_params_json = json.dumps(feature_params)
    metrics_json = json.dumps(metrics_dict)
    rl_params_json = json.dumps(rl_params)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO rl_models (model_name, model_blob, feature_params_json, metrics_json, rl_params_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (model_name, model_blob, feature_params_json, metrics_json, rl_params_json, datetime.now()))
        conn.commit()

def load_rl_model(model_name: str = "default_ppo"):
    """Carrega e desserializa um modelo de RL, seus parâmetros e métricas do banco de dados."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT model_blob, feature_params_json, metrics_json, rl_params_json FROM rl_models WHERE model_name = ?", (model_name,))
        row = cursor.fetchone()

        if row:
            model_blob, feature_params_json, metrics_json, rl_params_json = row
            buffer = io.BytesIO(model_blob)
            model = PPO.load(buffer)
            feature_params = json.loads(feature_params_json)
            metrics = json.loads(metrics_json)
            rl_params = json.loads(rl_params_json) if rl_params_json else {}
            return model, feature_params, metrics, rl_params
        else:
            return None, None, None, None

def save_model(username: str, model_object, metrics_dict: dict, feature_params: dict):
    """Serializa e salva o modelo, métricas e parâmetros de features do usuário no banco de dados."""
    model_buffer = io.BytesIO()
    joblib.dump(model_object, model_buffer)
    model_blob = model_buffer.getvalue()

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
        cursor.execute("SELECT model_blob, metrics_json, feature_params_json FROM user_models WHERE username = ?", (username,))
        row = cursor.fetchone()

        if row:
            model_blob, metrics_json, feature_params_json = row
            model_buffer = io.BytesIO(model_blob)
            model = joblib.load(model_buffer)
            metrics = json.loads(metrics_json)
            feature_params = json.loads(feature_params_json)
            metrics['feature_params'] = feature_params
            return model, metrics
        else:
            return None, None

def save_config(username: str, config: dict):
    """Salva a configuração do usuário no banco de dados."""
    model_params_json = json.dumps(config.get("model_params", {}))
    feature_params_json = json.dumps(config.get("feature_params", {}))
    rl_params_json = json.dumps(config.get("rl_params", {}))

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_configs (username, model_params_json, feature_params_json, rl_params_json)
            VALUES (?, ?, ?, ?)
        """, (username, model_params_json, feature_params_json, rl_params_json))
        conn.commit()

def load_config(username: str):
    """Carrega a configuração do usuário do banco de dados."""
    init_db() # Garante que a DB e as colunas estão atualizadas
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT model_params_json, feature_params_json, rl_params_json FROM user_configs WHERE username = ?", (username,))
        row = cursor.fetchone()

        if row:
            model_params = json.loads(row[0])
            feature_params = json.loads(row[1])
            rl_params = json.loads(row[2]) if row[2] else {}
            return {
                "model_params": model_params,
                "feature_params": feature_params,
                "rl_params": rl_params
            }
        else:
            # Valores Padrão
            return {
                "model_params": {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': -1, 'num_leaves': 31, 'reg_alpha': 0.0, 'reg_lambda': 0.0},
                "feature_params": {'sma_window_1': 10, 'sma_window_2': 30, 'rsi_window': 14, 'ema_window_1': 12, 'ema_window_2': 26, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'bollinger_window': 20, 'stochastic_window': 14},
                "rl_params": {'total_timesteps': 50000, 'learning_rate': 0.0003, 'gamma': 0.99, 'ent_coef': 0.1, 'holding_penalty': -0.1, 'trade_completion_bonus': 0.1, 'transaction_cost_pct': 0.001}
            }