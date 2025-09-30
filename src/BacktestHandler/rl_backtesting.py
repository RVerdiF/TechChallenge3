
import pandas as pd
import numpy as np

def run_rl_backtest(model, env):
    """
    Executa um backtest de uma estratégia de RL e calcula as métricas de desempenho.

    Args:
        model: O modelo de RL treinado (e.g., da stable-baselines3).
        env: A instância do ambiente de negociação (e.g., TradingEnv).

    Returns:
        dict: Dicionário com os resultados do backtest (métricas e histórico).
    """
    obs, info = env.reset()
    done = False
    
    portfolio_values = []
    initial_net_worth = env.initial_balance

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append({'date': env.df.loc[env.current_step-1, 'date'], 'value': env.net_worth})

    # ----- Cálculo das Métricas Finais -----
    portfolio_history = pd.DataFrame(portfolio_values).set_index('date')
    trades_history = pd.DataFrame(env.trades)

    # Retorno Total
    final_value = portfolio_history['value'].iloc[-1]
    total_return_pct = ((final_value / initial_net_worth) - 1) * 100
    
    # Buy and Hold
    buy_and_hold_shares = initial_net_worth / env.df['Close'].iloc[0]
    buy_and_hold_history = buy_and_hold_shares * env.df.set_index('date')['Close']
    buy_and_hold_history.name = "Buy & Hold"
    buy_and_hold_return_pct = ((buy_and_hold_history.iloc[-1] / initial_net_worth) - 1) * 100

    # Sharpe Ratio
    daily_returns = portfolio_history['value'].pct_change().dropna()
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max Drawdown
    portfolio_history['peak'] = portfolio_history['value'].cummax()
    portfolio_history['drawdown'] = (portfolio_history['value'] - portfolio_history['peak']) / portfolio_history['peak']
    max_drawdown = portfolio_history['drawdown'].min()

    # Métricas de Trade
    if not trades_history.empty and trades_history['profit'].notna().any():
        profitable_trades = trades_history[trades_history['profit'] > 0]
        win_rate = (len(profitable_trades) / len(trades_history[trades_history['profit'].notna()])) * 100
        total_trades = len(trades_history[trades_history['profit'].notna()])
    else:
        win_rate = 0
        total_trades = 0

    results = {
        "total_return_pct": total_return_pct,
        "buy_and_hold_return_pct": buy_and_hold_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "portfolio_history": portfolio_history['value'],
        "buy_and_hold_history": buy_and_hold_history,
        "trades_history": trades_history
    }

    return results
