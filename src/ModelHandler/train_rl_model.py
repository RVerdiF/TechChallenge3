
from stable_baselines3 import PPO
from src.DataHandler.data_handler import load_data
from src.DataHandler.model_db_handler import save_rl_model
from src.DataHandler import feature_engineering
from src.ReinforcementLearning.environment import TradingEnv
from src.BacktestHandler.rl_backtesting import run_rl_backtest
from src.LogHandler.log_config import get_logger

logger = get_logger(__name__)

def train_rl_model(feature_params: dict, rl_params: dict, start_date, end_date):
    """
    Carrega dados, cria features, treina um agente PPO e salva o modelo com suas métricas.
    """
    logger.info("Iniciando o treinamento do modelo de RL...")

    # 1. Carregar dados e filtrar pelo período
    logger.info(f"Carregando dados de {start_date} a {end_date} e criando features...")
    df_raw = load_data()
    if df_raw.empty:
        logger.warning("Banco de dados de preços está vazio. Abortando.")
        return
    
    df_raw = df_raw.loc[start_date:end_date]
    if df_raw.empty or len(df_raw) < 100:
        logger.warning("Dados insuficientes para o período selecionado. Abortando.")
        return

    df_features = feature_engineering.create_features(df_raw, params=feature_params)
    feature_cols = feature_engineering.get_feature_names(params=feature_params)
    df_features.dropna(inplace=True)
    if df_features.empty:
        logger.warning("Não há dados suficientes após a criação de features.")
        return

    # 2. Instanciar o ambiente
    logger.info("Instanciando o ambiente de negociação...")
    env = TradingEnv(
        df_features, 
        feature_cols=feature_cols,
        transaction_cost_pct=rl_params.get('transaction_cost_pct', 0.001),
        holding_penalty=rl_params.get('holding_penalty', -0.01),
        trade_completion_bonus=rl_params.get('trade_completion_bonus', 0.1)
    )

    # 3. Instanciar o agente PPO com hiperparâmetros
    logger.info(f"Instanciando o agente PPO com parâmetros: {rl_params}")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0, # verbose=1 imprime muitos logs, melhor desativar aqui
        learning_rate=rl_params.get('learning_rate', 0.0003),
        gamma=rl_params.get('gamma', 0.99),
        ent_coef=rl_params.get('ent_coef', 0.01)
    )

    # 4. Treinar o agente
    total_timesteps = rl_params.get('total_timesteps', 20000)
    logger.info(f"Iniciando o loop de treinamento para {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    logger.info("Treinamento do agente concluído.")

    # 5. Executar backtest para gerar métricas
    logger.info("Executando backtest para gerar métricas...")
    backtest_env = TradingEnv(
        df_features, 
        feature_cols=feature_cols,
        transaction_cost_pct=rl_params.get('transaction_cost_pct', 0.001),
        holding_penalty=rl_params.get('holding_penalty', -0.01),
        trade_completion_bonus=rl_params.get('trade_completion_bonus', 0.1)
    )
    metrics = run_rl_backtest(model, backtest_env)
    
    metrics_to_save = metrics.copy()
    metrics_to_save.pop('portfolio_history', None)
    metrics_to_save.pop('buy_and_hold_history', None)
    metrics_to_save.pop('trades_history', None)

    # 6. Salvar o modelo, parâmetros e métricas no banco de dados
    logger.info(f"Salvando o modelo com métricas: {metrics_to_save}")
    save_rl_model(model, feature_params=feature_params, metrics_dict=metrics_to_save, rl_params=rl_params)
    logger.info("Modelo salvo com sucesso!")

if __name__ == '__main__':
    default_feature_params = {
        'sma_window_1': 10, 'sma_window_2': 30, 'rsi_window': 14,
        'ema_window_1': 12, 'ema_window_2': 26, 'macd_fast': 12,
        'macd_slow': 26, 'macd_signal': 9, 'bollinger_window': 20,
        'stochastic_window': 14
    }
    default_rl_params = {
        'total_timesteps': 1000, # Reduzido para teste rápido
        'learning_rate': 0.0003,
        'gamma': 0.99
    }
    train_rl_model(feature_params=default_feature_params, rl_params=default_rl_params)
