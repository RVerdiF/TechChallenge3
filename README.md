# BTC Prediction Project

Projeto de previsão de preço do Bitcoin usando Machine Learning.

## Estrutura do Projeto

```
/btc_prediction_project
|-- requirements.txt      # Dependências do projeto
|-- data_api.py          # Coleta de dados via yfinance
|-- data_handler.py      # Gerenciamento de dados SQLite
|-- feature_engineering.py # Criação de features
|-- train_model.py       # Treinamento do modelo
|-- predict.py           # Previsões
|-- dashboard.py         # Interface Streamlit
|-- run_project.py       # Script de configuração
```

## Instalação

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Configuração Inicial
```bash
python run_project.py
```

### Dashboard
```bash
streamlit run dashboard.py
```

## Funcionalidades

- **Coleta de dados**: Histórico do Bitcoin via Yahoo Finance
- **Features**: SMA 10/30 dias, RSI 14 dias
- **Modelo**: LightGBM Classifier
- **Dashboard**: Interface web interativa
- **Previsão**: Tendência para o próximo dia (Alta/Queda)