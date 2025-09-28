import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import src.DataHandler.data_handler as data_handler
import src.DataHandler.feature_engineering as feature_engineering

def train_and_save_model(feature_params={}, model_params={}, model_path=None, metrics_path=None):
    """Treina o modelo de ML, salva em arquivo e salva as métricas."""
    
    # Carrega dados
    print("Carregando dados...")
    df = data_handler.load_data()
    
    if df.empty:
        print("Nenhum dado encontrado. Execute primeiro a coleta de dados.")
        return
    
    # Aplica engenharia de features
    print("Criando features...")
    df_features = feature_engineering.create_features(df, params=feature_params)
    
    if len(df_features) < 50:
        print("Dados insuficientes para treinamento.")
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
    feature_cols = base_features + indicator_features

    X = df_features[feature_cols]
    y = df_features['target']
    
    # Remove última linha (sem target)
    X = X[:-1]
    y = y[:-1]
    
    # Validação temporal
    tscv = TimeSeriesSplit(n_splits=3)
    accuracies = []
    f1_scores = []
    conf_matrix = np.zeros((2, 2))
    
    # Parâmetros do modelo com valores padrão
    lgbm_params = {
        'random_state': 42,
        'verbose': -1,
        **model_params
    }

    print("Treinando modelo...")
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LGBMClassifier(**lgbm_params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        conf_matrix += confusion_matrix(y_test, y_pred)
    
    # Treina modelo final com todos os dados
    final_model = LGBMClassifier(**lgbm_params)
    final_model.fit(X, y)
    
    if not model_path or not metrics_path:
        raise ValueError("model_path and metrics_path must be provided")

    # Salva modelo
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)

    # Calcula e salva métricas
    metrics = {
        "accuracy": np.mean(accuracies),
        "f1_score": np.mean(f1_scores),
        "features": feature_cols,
        "confusion_matrix": conf_matrix.tolist(),
        "feature_params": feature_params,
        "model_params": model_params
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Modelo treinado e salvo!")
    print(f"Acurácia média: {metrics['accuracy']:.3f}")
    print(f"F1-Score médio: {metrics['f1_score']:.3f}")

if __name__ == "__main__":
    train_and_save_model()