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
│   │   ├── data_handler.py     # Gerenciamento de dados SQLite
│   │   └── feature_engineering.py # Criação de features
│   ├── LogHandler/
│   │   └── log_config.py       # Configuração de logs
│   ├── ModelHandler/
│   │   ├── train_model.py      # Treinamento do modelo
│   │   └── predict.py          # Previsões
│   ├── Orchestration/
│   │   └── run_project.py      # Script de configuração inicial
│   ├── __init__.py             # Inicializador do pacote src
│   └── config.py               # Configurações do projeto (paths)
├── user_data/
│   └── Admin/
│       ├── lgbm_model.pkl      # Modelo treinado do usuário
│       └── metrics.json        # Métricas e parâmetros do modelo
├── .gitignore                  # Arquivos ignorados pelo Git
├── main.py                     # Interface Streamlit
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documentação
```

**Observações:**
- Os arquivos `btc_prices.db` e `users.db` são criados dinamicamente no diretório `src/DataHandler/`.
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

### Configuração Inicial
Execute uma única vez para configurar o projeto para o usuário padrão "Admin":
```bash
python -m src.Orchestration.run_project
```
Este comando irá:
- Coletar 3 anos de dados históricos do Bitcoin.
- Salvar os dados no banco SQLite (`btc_prices.db`).
- Treinar o modelo LightGBM com as features e parâmetros padrão.
- Salvar o modelo (`lgbm_model.pkl`) e as métricas (`metrics.json`) no diretório `user_data/Admin/`.

Para configurar para um novo usuário, use o argumento `--user`:
```bash
python -m src.Orchestration.run_project --user <username>
```

### Dashboard Interativo
```bash
streamlit run main.py
```
A aplicação irá pedir para criar um usuário e senha. Após o login, o dashboard será exibido.

## Funcionalidades

### Coleta de Dados
- **Fonte**: Yahoo Finance (yfinance)
- **Período**: Padrão de 3 anos, mas configurável no dashboard.
- **Dados**: Open, High, Low, Close, Volume
- **Armazenamento**: SQLite local (`btc_prices.db`)

### Engenharia de Features
- **Indicadores Técnicos**: SMA, RSI, EMA, MACD, Bandas de Bollinger, Oscilador Estocástico. Os parâmetros são configuráveis no dashboard.
- **Features de Data**: Mês, dia da semana, dia do mês.
- **Features de Lag**: Preço de fechamento de 7 e 30 dias atrás.
- **Features de Volume**: Variação percentual do volume.
- **Target**: Classificação binária (Alta/Queda do próximo dia).

### Modelo de Machine Learning
- **Algoritmo**: LightGBM Classifier
- **Validação**: Time Series Split (3 folds)
- **Métricas**: Accuracy e F1-Score
- **Previsão**: Tendência para o próximo dia (0=Queda, 1=Alta) com score de confiança.
- **Customização**: Os hiperparâmetros do modelo podem ser ajustados no dashboard.

### Dashboard Web
- **Autenticação**: Sistema de login e registro de usuários.
- **Gráfico interativo**: Histórico de preços com Plotly, com seleção de período.
- **Estatísticas**: Preço atual, variação diária, máximos/mínimos (30d).
- **Previsões**: Botão para gerar previsão do próximo dia com a confiança do modelo.
- **Treinamento Customizado**: Interface para ajustar parâmetros de features e do modelo, e treinar um novo modelo para o usuário logado.
- **Backtesting**: Página para simular uma estratégia de trading baseada no modelo treinado e visualizar os resultados.

## Fluxos da Aplicação

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
- Interface principal construída com Streamlit. Gerencia a navegação entre as páginas de login, dashboard, configurações e backtesting.

### `src/`
- **`config.py`**: Define os caminhos e constantes globais do projeto.

- **`ApiHandler/`**:
  - `data_api.py`: Interface com a API do Yahoo Finance para coleta de dados.

- **`AuthHandler/`**:
  - `auth.py`: Gerencia a autenticação de usuários, incluindo criação, login e armazenamento de senhas com hash em um banco de dados SQLite (`users.db`).

- **`BacktestHandler/`**:
  - `backtesting.py`: Contém a lógica para executar a simulação de backtesting, calculando o retorno da estratégia, Sharpe Ratio, Drawdown, etc.

- **`DataHandler/`**:
  - `data_handler.py`: Operações CRUD no banco de dados SQLite (`btc_prices.db`) para os preços do Bitcoin.
  - `feature_engineering.py`: Funções para calcular indicadores técnicos e outras features a partir dos dados brutos.

- **`LogHandler/`**:
  - `log_config.py`: Configuração centralizada do logger para o projeto.

- **`ModelHandler/`**:
  - `train_model.py`: Pipeline de treinamento do modelo. Inclui validação temporal (Time Series Split), salvamento do modelo treinado (`.pkl`) e das métricas (`.json`).
  - `predict.py`: Carrega um modelo treinado e a lista de features correspondente para fazer previsões em novos dados.

- **`Orchestration/`**:
  - `run_project.py`: Script para orquestrar a configuração inicial do projeto (coleta de dados e primeiro treinamento).

### `user_data/`
- Diretório que armazena os artefatos específicos de cada usuário. Cada usuário tem seu próprio subdiretório, contendo o modelo treinado e as métricas correspondentes.

## Dependências

- **pandas**: Manipulação de dados
- **yfinance**: API do Yahoo Finance
- **scikit-learn**: Métricas e validação
- **lightgbm**: Algoritmo de ML
- **joblib**: Serialização do modelo
- **streamlit**: Interface web
- **plotly**: Visualizações interativas
- **tqdm**: Barras de progresso

## Notas Técnicas

- **Validação Temporal**: O modelo usa `TimeSeriesSplit` para evitar *data leakage*, respeitando a natureza temporal dos dados.
- **Consistência de Features**: A lista de features é salva dinamicamente durante o treinamento e carregada durante a predição para garantir consistência e evitar erros.
- **Multi-usuário**: A arquitetura suporta múltiplos usuários, onde cada um pode treinar e usar seu próprio modelo.
