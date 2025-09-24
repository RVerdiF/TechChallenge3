# BTC Prediction Project

Projeto de previsão de preço do Bitcoin usando Machine Learning com arquitetura modular e dashboard interativo.

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
│   ├── ModelHandler/
│   │   ├── train_model.py       # Treinamento do modelo
│   │   ├── predict.py           # Previsões
│   │   ├── lgbm_model.pkl       # Modelo treinado
│   │   └── metrics.json         # Métricas e features do modelo
│   └── Orchestration/
│       └── run_project.py       # Script de configuração inicial
├── main.py                      # Interface Streamlit
├── requirements.txt             # Dependências do projeto
└── README.md                    # Documentação
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
python -m src.Orchestration.run_project
```
Este comando irá:
- Coletar 3 anos de dados históricos do Bitcoin
- Salvar os dados no banco SQLite
- Treinar o modelo LightGBM com as features atuais

### Dashboard Interativo
```bash
streamlit run main.py
```

## Funcionalidades

### Coleta de Dados
- **Fonte**: Yahoo Finance (yfinance)
- **Período**: 1095 dias (3 anos)
- **Dados**: Open, High, Low, Close, Volume
- **Armazenamento**: SQLite local

### Engenharia de Features
- **Indicadores Técnicos**: SMA (10, 30), RSI, EMA (12, 26), MACD, Bandas de Bollinger, Oscilador Estocástico.
- **Features de Data**: Mês, dia da semana, dia do mês.
- **Features de Lag**: Preço de fechamento de 7 e 30 dias atrás.
- **Features de Volume**: Variação percentual do volume.
- **Target**: Classificação binária (Alta/Queda do próximo dia).

### Modelo de Machine Learning
- **Algoritmo**: LightGBM Classifier
- **Validação**: Time Series Split (3 folds)
- **Métricas**: Accuracy e F1-Score
- **Previsão**: Tendência para o próximo dia (0=Queda, 1=Alta) com score de confiança.

### Dashboard Web
- **Gráfico interativo**: Histórico de preços com Plotly.
- **Estatísticas**: Preço atual, variação diária, máximos/mínimos (30d).
- **Métricas do Modelo**: Acurácia e F1-Score do modelo treinado.
- **Previsões**: Botão para gerar previsão do próximo dia com a confiança do modelo.
- **Controles**: Botões para atualizar os dados ou treinar o modelo diretamente pelo dashboard.

## Arquitetura Modular

### ApiHandler
- `data_api.py`: Interface com Yahoo Finance para coleta de dados.

### DataHandler  
- `data_handler.py`: Operações CRUD no banco SQLite.
- `feature_engineering.py`: Cálculo de indicadores técnicos e outras features.

### ModelHandler
- `train_model.py`: Pipeline de treinamento com validação temporal. Salva o modelo e um JSON com métricas e a lista de features.
- `predict.py`: Interface para fazer previsões com o modelo salvo, carregando a lista de features do JSON para evitar erros de inconsistência.

### Orchestration
- `run_project.py`: Script para orquestrar a configuração inicial do projeto.

## Dependências

- **pandas**: Manipulação de dados
- **yfinance**: API Yahoo Finance
- **scikit-learn**: Métricas e validação
- **lightgbm**: Algoritmo de ML
- **joblib**: Serialização do modelo
- **streamlit**: Interface web
- **plotly**: Visualizações interativas

## Notas Técnicas

- O modelo usa validação temporal para evitar data leakage.
- A lista de features é salva dinamicamente durante o treinamento e carregada durante a predição para garantir consistência.
- O dashboard permite a atualização de dados e o retreinamento do modelo de forma independente.