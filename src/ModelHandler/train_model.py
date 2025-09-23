import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import src.DataHandler.data_handler as data_handler
import src.DataHandler.feature_engineering as feature_engineering

def train_and_save_model():
    """Treina o modelo de ML e salva em arquivo."""
    
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
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_30', 'RSI']
    X = df_features[feature_cols]
    y = df_features['target']
    
    # Remove última linha (sem target)
    X = X[:-1]
    y = y[:-1]
    
    # Validação temporal
    tscv = TimeSeriesSplit(n_splits=3)
    accuracies = []
    f1_scores = []
    
    print("Treinando modelo...")
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LGBMClassifier(random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    
    # Treina modelo final com todos os dados
    final_model = LGBMClassifier(random_state=42, verbose=-1)
    final_model.fit(X, y)
    
    # Salva modelo
    joblib.dump(final_model, 'lgbm_model.pkl')
    
    print(f"Modelo treinado e salvo!")
    print(f"Acurácia média: {np.mean(accuracies):.3f}")
    print(f"F1-Score médio: {np.mean(f1_scores):.3f}")

if __name__ == "__main__":
    train_and_save_model()