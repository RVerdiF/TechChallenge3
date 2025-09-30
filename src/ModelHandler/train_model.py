import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import src.DataHandler.data_handler as data_handler
import src.DataHandler.feature_engineering as feature_engineering
import src.DataHandler.model_db_handler as model_db_handler
from src.LogHandler.log_config import get_logger

logger = get_logger(__name__)

def train_and_save_model(username: str, feature_params: dict, model_params: dict, start_date=None, end_date=None):
    """Treina o modelo de ML e o salva no banco de dados associado ao usuário."""
    
    if not username:
        raise ValueError("O nome de usuário deve ser fornecido para treinar e salvar um modelo.")

    # Carrega dados
    logger.info("Carregando dados para treinamento...")
    df = data_handler.load_data()
    
    if start_date and end_date:
        df = df.loc[start_date:end_date]
    
    if df.empty:
        logger.warning("Nenhum dado encontrado para o período selecionado. Abortando treinamento.")
        return
    
    # Aplica engenharia de features
    logger.info("Criando features...")
    df_features = feature_engineering.create_features(df, params=feature_params)
    
    if len(df_features) < 50: # Limite mínimo de dados
        logger.warning(f"Dados insuficientes para treinamento ({len(df_features)} registros). Abortando.")
        return
    
    # Gera a lista de features dinamicamente
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'month', 'day_of_week', 'day_of_month', 'close_7_days_ago', 'close_30_days_ago', 'volume_change_pct']
    sma_window_1 = feature_params.get('sma_window_1', 10)
    sma_window_2 = feature_params.get('sma_window_2', 30)
    ema_window_1 = feature_params.get('ema_window_1', 12)
    ema_window_2 = feature_params.get('ema_window_2', 26)

    indicator_features = [
        f'SMA_{sma_window_1}',
        f'SMA_{sma_window_2}',
        'RSI',
        f'EMA_{ema_window_1}',
        f'EMA_{ema_window_2}',
        'MACD',
        'MACD_signal',
        'Bollinger_Upper',
        'Bollinger_Lower',
        'Stochastic_K',
        'Stochastic_D'
    ]
    feature_cols = base_features + [feat for feat in indicator_features if feat in df_features.columns]

    X = df_features[feature_cols]
    y = df_features['target']
    
    # Remove a última linha (sem target) e linhas com NaN que podem ter sido geradas
    X = X.iloc[:-1].dropna()
    y = y.loc[X.index]
    
    # Validação temporal
    tscv = TimeSeriesSplit(n_splits=3) # Splits reduzidos para um treinamento mais rápido na UI
    accuracies = []
    f1_scores = []
    conf_matrix = np.zeros((2, 2))
    
    lgbm_params = {
        'random_state': 42,
        'verbose': -1,
        **model_params
    }

    logger.info(f"Iniciando treinamento do modelo para o usuário '{username}'...")
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LGBMClassifier(**lgbm_params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        conf_matrix += confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    # Treina modelo final com todos os dados
    logger.info("Treinando modelo final com todos os dados...")
    final_model = LGBMClassifier(**lgbm_params)
    final_model.fit(X, y)
    
    # Calcula e prepara métricas para salvar
    metrics = {
        "accuracy": np.mean(accuracies),
        "f1_score": np.mean(f1_scores),
        "features": feature_cols,
        "confusion_matrix": conf_matrix.tolist(),
        "model_params": model_params
    }
    
    # Salva modelo e métricas no banco de dados
    logger.info(f"Salvando modelo e métricas para o usuário '{username}' no banco de dados.")
    model_db_handler.save_model(
        username=username,
        model_object=final_model,
        metrics_dict=metrics,
        feature_params=feature_params
    )
    
    logger.info(f"Modelo para '{username}' treinado e salvo com sucesso!")
    logger.info(f"Acurácia média: {metrics['accuracy']:.3f}")
    logger.info(f"F1-Score médio: {metrics['f1_score']:.3f}")

if __name__ == "__main__":
    # Este bloco é para teste e agora requer um nome de usuário para ser executado.
    # Exemplo: train_and_save_model(username='test_user', feature_params={}, model_params={})
    pass
