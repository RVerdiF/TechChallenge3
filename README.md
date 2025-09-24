# BTC Prediction Project

Projeto de previsão de preço do Bitcoin usando Machine Learning com arquitetura modular.

## Estrutura do Projeto

```
/TC3
├── src/
│   ├── ApiHandler/
│   │   └── data_api.py          # Coleta de dados via yfinance
│   ├── DataHandler/
│   │   ├── data_handler.py      # Gerenciamento de dados SQLite
│   │   ├── feature_engineering.py # Criação de features
│   │   └── btc_prices.db        # Banco de dados SQLite
│   └── ModelHandler/
│       ├── train_model.py       # Treinamento do modelo
│       ├── predict.py           # Previsões
│       └── lgbm_model.pkl       # Modelo treinado
├── dashboard.py                 # Interface Streamlit
├── run_project.py              # Script de configuração inicial
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documentação
```

## Instalação

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Configuração Inicial
Execute uma única vez para configurar o projeto:
```bash
python run_project.py
```
Este comando irá:
- Coletar 3 anos de dados históricos do Bitcoin
- Salvar os dados no banco SQLite
- Treinar o modelo LightGBM

### Dashboard Interativo
```bash
streamlit run dashboard.py
```

## Funcionalidades

### Coleta de Dados
- **Fonte**: Yahoo Finance (yfinance)
- **Período**: Configurável (padrão: últimos 365 dias)
- **Dados**: Open, High, Low, Close, Volume
- **Armazenamento**: SQLite local

### Engenharia de Features
- **SMA 10 dias**: Média móvel simples de 10 períodos
- **SMA 30 dias**: Média móvel simples de 30 períodos  
- **RSI 14 dias**: Índice de Força Relativa
- **Target**: Classificação binária (Alta/Queda do próximo dia)

### Modelo de Machine Learning
- **Algoritmo**: LightGBM Classifier
- **Validação**: Time Series Split (3 folds)
- **Métricas**: Accuracy e F1-Score
- **Previsão**: Tendência para o próximo dia (0=Queda, 1=Alta)

### Dashboard Web
- **Gráfico interativo**: Histórico de preços com Plotly
- **Métricas em tempo real**: Preço atual, variação diária, máximos/mínimos
- **Previsões**: Botão para gerar previsão do próximo dia
- **Atualização de dados**: Botão para buscar dados mais recentes

## Arquitetura Modular

### ApiHandler
- `data_api.py`: Interface com Yahoo Finance para coleta de dados

### DataHandler  
- `data_handler.py`: Operações CRUD no banco SQLite
- `feature_engineering.py`: Cálculo de indicadores técnicos

### ModelHandler
- `train_model.py`: Pipeline de treinamento com validação temporal
- `predict.py`: Interface para fazer previsões com modelo salvo

## Dependências

- **pandas**: Manipulação de dados
- **yfinance**: API Yahoo Finance
- **scikit-learn**: Métricas e validação
- **lightgbm**: Algoritmo de ML
- **joblib**: Serialização do modelo
- **streamlit**: Interface web
- **plotly-express**: Visualizações interativas

## Notas Técnicas

- O modelo usa validação temporal para evitar data leakage
- Dados são atualizados incrementalmente no SQLite
- Features são calculadas dinamicamente
- Interface responsiva com Streamlit