
import pandas as pd
import numpy as np
from tqdm import tqdm

def run_backtest(df, model, feature_cols, initial_capital=1000.0, start_date=None, end_date=None):
    """
    Executa o backtesting de uma estratégia de trading baseada em um modelo.

    Args:
        df (pd.DataFrame): DataFrame com dados históricos, incluindo preços e features.
        model: Modelo de machine learning treinado com método predict.
        feature_cols (list): Lista de colunas de features a serem usadas pelo modelo.
        initial_capital (float): Capital inicial para a simulação.

    Returns:
        dict: Um dicionário contendo as métricas de resultados do backtest.
        pd.DataFrame: Um DataFrame com o histórico de operações.
    """
    
    capital = initial_capital
    btc_held = 0.0
    portfolio_values = []
    trades = []
    position = None # None, 'long'

    # Garante que o dataframe de entrada não seja modificado
    df_backtest = df.copy()

    if start_date and end_date:
        df_backtest = df_backtest.loc[start_date:end_date]

    # Remove a última linha que não tem target e não pode ser usada para predição
    if 'target' in df_backtest.columns:
        df_backtest = df_backtest[df_backtest['target'].notna()]

    for i in tqdm(range(len(df_backtest) - 1), desc="Executando Backtest"):
        current_row = df_backtest.iloc[i]
        current_price = current_row['Close']
        
        # Prepara os dados para o modelo
        X_live = current_row[feature_cols].to_frame().T
        
        # Gera o sinal de trading para o *próximo* dia
        signal = model.predict(X_live)[0]

        # Lógica de execução da estratégia
        if signal == 1 and position is None: # Sinal de ALTA e não temos posição
            # Compra BTC
            btc_bought = capital / current_price
            capital = 0.0
            btc_held = btc_bought
            position = 'long'
            trades.append({
                'date_buy': current_row.name,
                'price_buy': current_price,
                'date_sell': None,
                'price_sell': None,
                'profit': None
            })
        elif signal == 0 and position == 'long': # Sinal de QUEDA e temos posição
            # Vende BTC
            capital = btc_held * current_price
            btc_held = 0.0
            position = None
            # Atualiza a última operação na lista de trades
            if trades:
                last_trade = trades[-1]
                last_trade['date_sell'] = current_row.name
                last_trade['price_sell'] = current_price
                last_trade['profit'] = last_trade['price_sell'] - last_trade['price_buy']

        # Calcula o valor do portfólio no final do dia
        current_portfolio_value = capital + (btc_held * current_price)
        portfolio_values.append({'date': current_row.name, 'value': current_portfolio_value})

    # Se a última operação ainda estiver aberta, fecha na última cotação
    if position == 'long':
        last_price = df_backtest.iloc[-1]['Close']
        capital = btc_held * last_price
        btc_held = 0.0
        if trades:
            last_trade = trades[-1]
            last_trade['date_sell'] = df_backtest.iloc[-1].name
            last_trade['price_sell'] = last_price
            last_trade['profit'] = last_trade['price_sell'] - last_trade['price_buy']

    # ----- Cálculo das Métricas Finais -----
    portfolio_history = pd.DataFrame(portfolio_values).set_index('date')
    trades_history = pd.DataFrame(trades)

    # Total Return
    total_return_pct = ((portfolio_history['value'].iloc[-1] / initial_capital) - 1) * 100
    
    # Buy and Hold
    buy_and_hold_shares = initial_capital / df_backtest['Close'].iloc[0]
    buy_and_hold_history = buy_and_hold_shares * df_backtest['Close']
    buy_and_hold_history.name = "Buy & Hold"
    buy_and_hold_return_pct = ((buy_and_hold_history.iloc[-1] / initial_capital) - 1) * 100

    # Sharpe Ratio
    daily_returns = portfolio_history['value'].pct_change().dropna()
    if daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max Drawdown
    portfolio_history['peak'] = portfolio_history['value'].cummax()
    portfolio_history['drawdown'] = (portfolio_history['value'] - portfolio_history['peak']) / portfolio_history['peak']
    max_drawdown = portfolio_history['drawdown'].min()

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
        "portfolio_history": portfolio_history,
        "buy_and_hold_history": buy_and_hold_history
    }

    return results, trades_history
