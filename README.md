# BTC Prediction Project

Projeto de previsão de preço do Bitcoin usando Machine Learning com arquitetura modular e dashboard interativo.

## Estrutura do Projeto

```
/Tech_Challenge_3
├── .devcontainer/              # Configurações do Dev Container
├── .git/                       # Repositório Git
├── .venv/                      # Ambiente virtual Python
├── src/
│   ├── ApiHandler/
│   │   └── data_api.py         # Coleta de dados via yfinance
│   ├── AuthHandler/
│   │   └── auth.py             # Gerenciamento de autenticação de usuários
│   ├── BacktestHandler/
│   │   └── backtesting.py      # Lógica para backtesting de estratégias
│   ├── DataHandler/
│   │   ├── data_handler.py     # Gerenciamento do DB de preços (btc_prices.db)
│   │   ├── model_db_handler.py # Gerenciamento do DB de modelos (models.db)
│   │   └── feature_engineering.py # Criação de features
│   ├── LogHandler/
│   │   └── log_config.py       # Configuração de logs
│   ├── ModelHandler/
│   │   ├── train_model.py      # Treinamento do modelo
│   │   └── predict.py          # Previsões
│   ├── Orchestration/
│   │   ├── run_project.py      # Script de configuração inicial (obsoleto)
│   │   └── update_scheduler.py # Lógica para atualização diária de dados
│   ├── __init__.py             # Inicializador do pacote src
│   └── config.py               # Configurações do projeto (paths)
├── .gitignore                  # Arquivos ignorados pelo Git
├── main.py                     # Interface Streamlit
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documentação
```

**Observações:**
- Os bancos de dados `btc_prices.db`, `users.db` e `models.db` são criados dinamicamente no diretório `src/DataHandler/`.
- Os diretórios `__pycache__` são criados automaticamente pelo Python.

## Instalação

1. Clone o repositório
2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate    # Windows
```
3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

**Para iniciar o dashboard interativo:**
```bash
streamlit run main.py
```
A aplicação irá iniciar e apresentar uma tela de login. Você pode criar um novo usuário e, após o login, o dashboard será exibido. A primeira carga de dados é feita automaticamente em segundo plano.

## Funcionalidades

### Coleta de Dados
- **Fonte**: Yahoo Finance (yfinance)
- **Atualização Automática**: Os dados são atualizados diariamente em segundo plano para incluir o dia mais recente.
- **Armazenamento**: SQLite local (`btc_prices.db`).

### Engenharia de Features
- **Indicadores Técnicos**: SMA, RSI, EMA, MACD, Bandas de Bollinger, Oscilador Estocástico.
- **Target**: Classificação binária (Alta/Queda do próximo dia).

### Modelo de Machine Learning
- **Algoritmo**: LightGBM Classifier
- **Validação**: Time Series Split (3 folds)
- **Métricas**: Accuracy e F1-Score
- **Armazenamento**: O modelo treinado, métricas e parâmetros são salvos em um banco de dados SQLite, associados à conta do usuário.

### Dashboard Web
- **Autenticação**: Sistema de login e registro de usuários.
- **Atualização Automática**: Os dados de preço são atualizados diariamente em background.
- **Gráfico interativo**: Histórico de preços com Plotly, com seleção de período.
- **Previsões**: Botão para gerar previsão do próximo dia com a confiança do modelo do usuário.
- **Treinamento Customizado**: Interface para ajustar parâmetros de features e do modelo, e treinar um novo modelo para o usuário logado. Os parâmetros são salvos no banco de dados para futuras sessões.
- **Backtesting**: Página para simular uma estratégia de trading baseada no modelo treinado.

## Fluxos da Aplicação

(Os diagramas de fluxo fornecem uma visão geral da lógica da aplicação. Note que o armazenamento, antes baseado em arquivos, agora é centralizado em bancos de dados SQLite.)

### Fluxo Principal
![Fluxo Principal da Aplicação](imgs/Main%20Application%20Flow.jpg)

### Fluxo de Atualização de Dados (Dashboard)
![Fluxo de Atualização de Dados](imgs/Data%20Update%20Flow%20(Dashboard).jpg)

### Fluxo de Predição (Dashboard)
![Fluxo de Predição](imgs/Prediction%20Flow%20(Dashboard).jpg)

### Fluxo de Treinamento (Configurações)
![Fluxo de Treinamento](imgs/Training%20Flow%20(Settings).jpg)

### Fluxo de Backtesting
![Fluxo de Backtesting](imgs/Backtesting%20Flow%20(Backtesting).jpg)

## Arquitetura Modular

### `main.py`
- Interface principal com Streamlit. Gerencia a navegação, estado da sessão e inicialização de tarefas em segundo plano (atualização de dados).

### `src/`
- **`ApiHandler/`**: Coleta de dados externos.
- **`AuthHandler/`**: Gerencia a autenticação de usuários em seu próprio banco de dados (`users.db`).
- **`BacktestHandler/`**: Contém a lógica para executar a simulação de backtesting.
- **`DataHandler/`**:
  - `data_handler.py`: Gerencia o banco de dados de preços (`btc_prices.db`) e metadados (ex: data da última atualização).
  - `model_db_handler.py`: Gerencia o banco de dados de modelos e configurações dos usuários (`models.db`).
  - `feature_engineering.py`: Funções para calcular indicadores técnicos.
- **`LogHandler/`**: Configuração centralizada do logger.
- **`ModelHandler/`**:
  - `train_model.py`: Pipeline de treinamento que salva o modelo e métricas no banco de dados via `model_db_handler`.
  - `predict.py`: Carrega um modelo do banco de dados para fazer previsões.
- **`Orchestration/`**:
  - `update_scheduler.py`: Contém a lógica da tarefa em segundo plano que verifica e dispara a atualização diária dos dados.

## Dependências

- **pandas**: Manipulação de dados
- **yfinance**: API do Yahoo Finance
- **scikit-learn**: Métricas e validação
- **lightgbm**: Algoritmo de ML
- **joblib**: Serialização do modelo
- **streamlit**: Interface web
- **plotly**: Visualizações interativas

## Notas Técnicas

- **Multi-usuário**: A arquitetura suporta múltiplos usuários, onde cada um pode treinar, salvar e usar seu próprio modelo e configurações, persistidos em um banco de dados.
- **Tarefas em Segundo Plano**: A atualização diária de dados é executada em uma thread separada, garantindo que a interface do usuário não seja bloqueada.