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

MODEL_PATH = Path('src/ModelHandler/lgbm_model.pkl')
METRICS_PATH = Path('src/ModelHandler/metrics.json')

def train_and_save_model():
    """Treina o modelo de ML, salva em arquivo e salva as métricas."""
    
    # Carrega dados
    print("Carregando dados...")
    df = data_handler.load_data()
    
    if df.empty:
        print("Nenhum dado encontrado. Execute primeiro a coleta de dados.")
        return
    
    # Aplica engenharia de features
    print("Criando features...")
    df_features = feature_engineering.create_features(df)
    
    if len(df_features) < 50:
        print("Dados insuficientes para treinamento.")
        return
    
    # Prepara dados para ML
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_30', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'Bollinger_Upper', 'Bollinger_Lower', 'Stochastic_K', 'Stochastic_D', 'month', 'day_of_week', 'day_of_month', 'close_7_days_ago', 'close_30_days_ago', 'volume_change_pct']
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
    
    print("Treinando modelo...")
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LGBMClassifier(random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        conf_matrix += confusion_matrix(y_test, y_pred)
    
    # Treina modelo final com todos os dados
    final_model = LGBMClassifier(random_state=42, verbose=-1)
    final_model.fit(X, y)
    
    # Salva modelo
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    # Calcula e salva métricas
    metrics = {
        "accuracy": np.mean(accuracies),
        "f1_score": np.mean(f1_scores),
        "features": feature_cols,
        "confusion_matrix": conf_matrix.tolist()
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Modelo treinado e salvo!")
    print(f"Acurácia média: {metrics['accuracy']:.3f}")
    print(f"F1-Score médio: {metrics['f1_score']:.3f}")

if __name__ == "__main__":
    train_and_save_model()